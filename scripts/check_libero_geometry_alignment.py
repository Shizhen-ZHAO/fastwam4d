"""Check raw demonstration image orientation against a simulator state replay."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from libero.libero import benchmark
from experiments.libero.libero_utils import get_libero_env, get_libero_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/homes/zhaoshizhen/datasets/libero/official_hf/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5")
    parser.add_argument("--output", default="outputs/libero_geometry_alignment.json")
    args = parser.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_id = next(i for i in range(suite.n_tasks) if "table_center" in suite.get_task(i).name)
    task = suite.get_task(task_id)
    env, _ = get_libero_env(task, resolution=128, seed=42)
    try:
        env.reset()
        with h5py.File(args.data, "r") as handle:
            demo = handle["data/demo_0"]
            obs = env.set_init_state(demo["states"][20])
            current = get_libero_image(obs)
            result = {"task_id": task_id, "source": args.data, "comparisons": {}}
            for source_key, key in (("agentview_rgb", "image"), ("eye_in_hand_rgb", "wrist_image")):
                saved = demo[f"obs/{source_key}"][20]
                variants = {"none": saved, "rotate_180": saved[::-1, ::-1],
                            "flip_vertical": saved[::-1], "flip_horizontal": saved[:, ::-1]}
                errors = {name: float(np.mean((array.astype(np.float32) - current[key]) ** 2))
                          for name, array in variants.items()}
                result["comparisons"][key] = {"mse_to_policy_observation": errors, "best": min(errors, key=errors.get)}
            result["proprio_position_max_error"] = float(np.abs(demo["obs/ee_pos"][20] - obs["robot0_eef_pos"]).max())
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
