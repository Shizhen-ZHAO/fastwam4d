"""严格比较所有rank/所有microbatch；任意缺失、loss或输入/参数差异均失败。"""
import argparse
import json
import math
from pathlib import Path


START_KEYS = ['rank', 'contract', 'environment', 'distributed_type', 'deepspeed_config',
              'initial_parameters', 'parameter_order', 'vae_scale_effective', 'rng',
              'dataset_stats_sha256', 'parameter_hash_every', 'initial_global_step']
STEP_KEYS = ['event', 'microbatch', 'global_step_before', 'epoch', 'batch_in_epoch',
             'sample', 'rng_before', 'inputs', 'noise_calls', 'rng_after_forward',
             'loss', 'metrics', 'lr_before', 'update_boundary', 'optimizer_step_was_skipped',
             'parameters_after_update', 'rng_after_backward']


def first_difference(a, b, location):
    if type(a) is not type(b):
        return {'location': location, 'left': a, 'right': b}
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return {'location': location + '.keys', 'left': sorted(a), 'right': sorted(b)}
        for key in a:
            result = first_difference(a[key], b[key], location + '.' + key)
            if result:
                return result
    elif isinstance(a, list):
        if len(a) != len(b):
            return {'location': location + '.length', 'left': len(a), 'right': len(b)}
        for index, (x, y) in enumerate(zip(a, b)):
            result = first_difference(x, y, location + f'[{index}]')
            if result:
                return result
    elif a != b:
        result = {'location': location, 'left': a, 'right': b}
        if isinstance(a, (float, int)) and not isinstance(a, bool):
            result['absolute_difference'] = abs(a - b)
        return result
    return None


def read_trace(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) < 3 or rows[0].get('event') != 'start' or rows[-1].get('event') != 'end':
        raise ValueError(f'Incomplete trace: {path}')
    start, steps, end = rows[0], rows[1:-1], rows[-1]
    expected_updates = start['contract']['max_steps'] - start['initial_global_step']
    if not end.get('completed') or end['global_step'] != start['contract']['max_steps']:
        raise ValueError(f'Training did not reach expected max_steps: {path}')
    if len(steps) != end['microbatches'] or len(steps) != expected_updates * start['contract']['gradient_accumulation_steps']:
        raise ValueError(f'Missing or unexpected microbatches: {path}')
    if sum(bool(row['update_boundary']) for row in steps) != expected_updates:
        raise ValueError(f'Incorrect update count: {path}')
    for index, row in enumerate(steps, 1):
        if row.get('event') != 'microbatch' or row['microbatch'] != index or not math.isfinite(row['loss']):
            raise ValueError(f'Invalid loss record: {path}:{index + 1}')
    return start, steps, end


def compare_directories(left_dir, right_dir):
    left_files = {p.name: p for p in left_dir.glob('rank_*.jsonl')}
    right_files = {p.name: p for p in right_dir.glob('rank_*.jsonl')}
    if not left_files or left_files.keys() != right_files.keys():
        raise ValueError('Rank trace files are missing or do not match')
    local_losses = {}
    total_microbatches = 0
    for name in sorted(left_files):
        a_start, a_steps, a_end = read_trace(left_files[name])
        b_start, b_steps, b_end = read_trace(right_files[name])
        expected = {f'rank_{rank:05d}.jsonl' for rank in range(a_start['contract']['world_size'])}
        if set(left_files) != expected:
            raise ValueError('Incomplete rank coverage')
        if name != f"rank_{a_start['rank']:05d}.jsonl" or name != f"rank_{b_start['rank']:05d}.jsonl":
            raise ValueError('Trace filename and recorded rank do not match')
        for key in START_KEYS:
            difference = first_difference(a_start[key], b_start[key], name + '.start.' + key)
            if difference:
                return {'passed': False, 'first_difference': difference}
        for index, (a, b) in enumerate(zip(a_steps, b_steps), 1):
            # 先比较输入/随机数，再比较loss，优先报告导致差异的上游字段。
            ordered = [key for key in STEP_KEYS if key in a]
            ordered += [key for key in a if key not in ordered]
            ordered_a = {key: a[key] for key in ordered}
            difference = first_difference(ordered_a, b, name + f'.microbatch[{index}]')
            if difference:
                return {'passed': False, 'first_difference': difference}
            local_losses.setdefault(a['global_step_before'] + 1, []).append(a['loss'])
        difference = first_difference(a_end, b_end, name + '.end')
        if difference:
            return {'passed': False, 'first_difference': difference}
        total_microbatches += len(a_steps)
    return {'passed': True, 'comparison': 'exact; zero tolerance', 'ranks': len(left_files),
            'microbatch_records_compared': total_microbatches, 'max_loss_absolute_difference': 0.0,
            'updates': [{'step': step, 'mean_local_microbatch_loss': sum(values) / len(values)}
                        for step, values in sorted(local_losses.items())]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = compare_directories(args.reference, args.target)
    except (ValueError, KeyError, OSError) as exc:
        result = {'passed': False, 'error': str(exc)}
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
