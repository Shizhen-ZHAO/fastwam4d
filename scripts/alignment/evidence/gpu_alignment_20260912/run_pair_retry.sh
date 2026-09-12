#!/usr/bin/env bash
# Robust phase runner for sporadic workers>0 hangs on this PPU/NFS stack.
# Usage: run_pair_retry.sh <phase-name> <run_pair args...>
#
# Stall classification via py-spy probe when the log goes quiet:
#   - all trainer mains inside torch.save/serialization (NFS checkpoint write) -> keep waiting
#   - mains in allreduce / dataloader queue waits (fork-lock hang)             -> kill and retry
# A fresh try_<N> subroot is used per attempt; success marker written at the end.
set -uo pipefail
PHASE_NAME="${1:?phase name}"; shift
REVIEW_ROOT_V=/ossfs/workspace/alignment_results/20260912_001614
QUIET_SECS="${QUIET_SECS:-420}"     # log-quiet threshold that triggers a py-spy probe
SAVE_GRACE_SECS="${SAVE_GRACE_SECS:-2400}"  # max continued quiet while in save state
MAX_TRIES="${MAX_TRIES:-15}"
PYSPY_DIR="$REVIEW_ROOT_V/${PHASE_NAME}_pyspy"
mkdir -p "$PYSPY_DIR"

probe_and_classify() {
  # Returns 0 = true hang (kill+retry), 1 = benign (save/teardown/transition, keep waiting)
  # Fast parallel probe; a desync hang shows >=75% of trainer mains in collective/queue waits.
  local stamp probe_dir mains saving waiting threshold
  stamp=$(date +%H%M%S)
  probe_dir="$PYSPY_DIR/probe_${stamp}"
  mkdir -p "$probe_dir"
  for pid in $(pgrep -f "train_with_trace.py"); do
    py-spy dump --pid "$pid" > "$probe_dir/${pid}.txt" 2>/dev/null &
  done
  wait
  mains=$(grep -l '"MainThread"' "$probe_dir"/*.txt 2>/dev/null | wc -l)
  saving=$(grep -l '"MainThread"' "$probe_dir"/*.txt 2>/dev/null | xargs -r grep -lE "serialization|torch_checkpoint|\.save|save_mp4" 2>/dev/null | wc -l)
  waiting=$(grep -l '"MainThread"' "$probe_dir"/*.txt 2>/dev/null | xargs -r grep -lE "allreduce|_try_get_data|ProcessGroupNCCL|nccl" 2>/dev/null | wc -l)
  threshold=$(( (mains * 3 + 3) / 4 ))
  echo "[probe $stamp] mains=$mains saving=$saving waiting=$waiting threshold=$threshold" >> "$REVIEW_ROOT_V/${PHASE_NAME}_runner.log"
  [ "$mains" -ge 8 ] && [ "$waiting" -ge "$threshold" ] && return 0
  return 1
}

for try in $(seq 1 "$MAX_TRIES"); do
  ROOT="$REVIEW_ROOT_V/${PHASE_NAME}_try${try}"
  LOG="$REVIEW_ROOT_V/${PHASE_NAME}_try${try}.log"
  echo "[runner] attempt $try -> $ROOT" >> "$REVIEW_ROOT_V/${PHASE_NAME}_runner.log"
  bash /ossfs/workspace/alignment_results/20260912_001614/run_phase_inner.sh "$ROOT" "$@" > "$LOG" 2>&1 &
  RUN_PID=$!
  LAST_SIZE=-1; LAST_CHANGE=$(date +%s); SAVE_SINCE=0
  while kill -0 "$RUN_PID" 2>/dev/null; do
    sleep 30
    SIZE=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    if [ "$SIZE" != "$LAST_SIZE" ]; then LAST_SIZE=$SIZE; LAST_CHANGE=$NOW; fi
    QUIET=$((NOW - LAST_CHANGE))
    if [ "$QUIET" -ge "$QUIET_SECS" ]; then
      if pgrep -f "train_with_trace.py" > /dev/null 2>&1; then
        if probe_and_classify; then
          echo "[runner] attempt $try TRUE HANG at $(date +%H:%M:%S); killing" >> "$REVIEW_ROOT_V/${PHASE_NAME}_runner.log"
          pkill -9 -f "train_with_trace.py" 2>/dev/null || true
          pkill -9 -f "accelerate.commands.launch" 2>/dev/null || true
          kill -9 "$RUN_PID" 2>/dev/null || true
          break
        else
          # benign (e.g. NFS checkpoint save): keep waiting, but bound it
          if [ $((NOW - LAST_CHANGE)) -ge "$SAVE_GRACE_SECS" ]; then
            echo "[runner] attempt $try save-grace exceeded; killing" >> "$REVIEW_ROOT_V/${PHASE_NAME}_runner.log"
            pkill -9 -f "train_with_trace.py" 2>/dev/null || true
            pkill -9 -f "accelerate.commands.launch" 2>/dev/null || true
            kill -9 "$RUN_PID" 2>/dev/null || true
            break
          fi
        fi
      else
        LAST_CHANGE=$NOW   # no trainers: teardown in progress, keep waiting
      fi
    fi
  done
  # Collect the real exit status; wait works whether or not the run already exited.
  wait "$RUN_PID"; RC=$?
  echo "[runner] attempt $try ended rc=$RC" >> "$REVIEW_ROOT_V/${PHASE_NAME}_runner.log"
  if [ "$RC" = 0 ]; then
    echo "$ROOT" > "$REVIEW_ROOT_V/${PHASE_NAME}_SUCCESS_ROOT"
    exit 0
  fi
  # Distinguish hang-class failures (retry) from real code errors (stop).
  # Hang signatures: NCCL watchdog, SIGABRT, elastic ChildFailedError with negative exitcodes.
  if ! tail -400 "$LOG" 2>/dev/null | grep -qE "watchdog|Watchdog|SIGABRT|ChildFailedError|exitcode: -"; then
    if tail -400 "$LOG" 2>/dev/null | grep -qE "Traceback"; then
      echo "[runner] attempt $try REAL ERROR; stopping for inspection" >> "$REVIEW_ROOT_V/${PHASE_NAME}_runner.log"
      exit "$RC"
    fi
  fi
  # Wait for trainers to exit and PPU memory to drain before the next attempt.
  for _ in $(seq 1 30); do
    pgrep -f "train_with_trace.py" > /dev/null 2>&1 || break
    sleep 10
  done
  pkill -9 -f "train_with_trace.py" 2>/dev/null || true
  for _ in $(seq 1 30); do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
    [ "${MEM:-0}" -lt 2000 ] && break
    sleep 10
  done
done
echo "[runner] exhausted $MAX_TRIES tries" >> "$REVIEW_ROOT_V/${PHASE_NAME}_runner.log"
exit 99
