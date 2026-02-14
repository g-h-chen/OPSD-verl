import argparse
import json

import pandas as pd


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return repr(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a few parquet samples.")
    parser.add_argument(
        "--path",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/train.parquet",
        help="Parquet dataset path.",
    )
    parser.add_argument("--num", type=int, default=20, help="Number of samples to print.")
    args = parser.parse_args()

    df = pd.read_parquet(args.path)
    for idx, row in df.head(args.num).iterrows():
        record = row.to_dict()
        print(f"--- sample {idx} ---")
        print(json.dumps(record, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

