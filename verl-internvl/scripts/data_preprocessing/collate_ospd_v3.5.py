import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


_COMBINED_OPTIMAL1_WEIGHTS = {
    (0, 0): 6118,
    (0, 1): 756,
    (1, 0): 1936,
    (1, 1): 990,
}


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_answer_acc_mode(mode: Optional[str]) -> Optional[Tuple[int, int]]:
    if not mode:
        return None
    if mode in {"balanced", "combined_optimal1"}:
        return None
    match = re.match(r"^wh([01])_woh([01])$", mode)
    if not match:
        raise ValueError(f"Invalid answer_acc_mode: {mode}")
    return int(match.group(1)), int(match.group(2))


def _answer_acc_key(record: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    extra_info = record.get("extra_info") or {}
    if "w_h_c" not in extra_info or "wo_h_c" not in extra_info:
        return None
    wh_bit = 1 if bool(extra_info.get("w_h_c")) else 0
    woh_bit = 1 if bool(extra_info.get("wo_h_c")) else 0
    return wh_bit, woh_bit


def _match_answer_acc_mode(record: Dict[str, Any], mode: Optional[Tuple[int, int]]) -> bool:
    if mode is None:
        return True
    key = _answer_acc_key(record)
    if key is None:
        return False
    return key == mode


def _collect_records(
    input_path: str,
    max_samples: int,
    answer_acc_mode: Optional[Tuple[int, int]],
    desc: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    records: List[Dict[str, Any]] = []
    scanned = 0
    kept = 0
    for record in tqdm(_iter_jsonl(input_path), desc=desc):
        scanned += 1
        if max_samples and kept >= max_samples:
            break
        if not _match_answer_acc_mode(record, answer_acc_mode):
            continue
        records.append(record)
        kept += 1
    return records, scanned, kept


def _collect_records_balanced(
    input_path: str,
    max_samples: int,
    desc: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    mode_keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    mode_records: Dict[Tuple[int, int], List[Dict[str, Any]]] = {key: [] for key in mode_keys}
    scanned = 0

    if max_samples:
        base = max_samples // 4
        remainder = max_samples % 4
        targets = {key: base for key in mode_keys}
        for idx, key in enumerate(mode_keys):
            if idx < remainder:
                targets[key] += 1
        kept = 0
        for record in tqdm(_iter_jsonl(input_path), desc=desc):
            scanned += 1
            key = _answer_acc_key(record)
            if key not in targets:
                continue
            if len(mode_records[key]) >= targets[key]:
                if all(len(mode_records[m]) >= targets[m] for m in mode_keys):
                    break
                continue
            mode_records[key].append(record)
            kept += 1
            if all(len(mode_records[m]) >= targets[m] for m in mode_keys):
                break
        records: List[Dict[str, Any]] = []
        for key in mode_keys:
            records.extend(mode_records[key])
        return records, scanned, len(records)

    for record in tqdm(_iter_jsonl(input_path), desc=desc):
        scanned += 1
        key = _answer_acc_key(record)
        if key not in mode_records:
            continue
        mode_records[key].append(record)
    min_count = min((len(mode_records[key]) for key in mode_keys), default=0)
    records = []
    for key in mode_keys:
        records.extend(mode_records[key][:min_count])
    return records, scanned, len(records)


def _normalize_weights(weights: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0:
        return {key: 0.0 for key in weights}
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def _compute_ratio_targets(
    max_samples: int,
    ratios: Dict[Tuple[int, int], float],
    mode_keys: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], int]:
    if max_samples <= 0:
        return {key: 0 for key in mode_keys}
    raw = {key: ratios.get(key, 0.0) * max_samples for key in mode_keys}
    targets = {key: int(raw[key]) for key in mode_keys}
    remainder = max_samples - sum(targets.values())
    if remainder > 0:
        fractional = sorted(
            ((raw[key] - targets[key], key) for key in mode_keys),
            reverse=True,
        )
        for _, key in fractional[:remainder]:
            targets[key] += 1
    return targets


def _collect_records_weighted(
    input_path: str,
    max_samples: int,
    weights: Dict[Tuple[int, int], float],
    desc: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    mode_keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    mode_records: Dict[Tuple[int, int], List[Dict[str, Any]]] = {key: [] for key in mode_keys}
    scanned = 0
    ratios = _normalize_weights(weights)

    if max_samples:
        targets = _compute_ratio_targets(max_samples, ratios, mode_keys)
        kept = 0
        for record in tqdm(_iter_jsonl(input_path), desc=desc):
            scanned += 1
            key = _answer_acc_key(record)
            if key not in targets:
                continue
            if len(mode_records[key]) >= targets[key]:
                if all(len(mode_records[m]) >= targets[m] for m in mode_keys):
                    break
                continue
            mode_records[key].append(record)
            kept += 1
            if all(len(mode_records[m]) >= targets[m] for m in mode_keys):
                break
        records: List[Dict[str, Any]] = []
        for key in mode_keys:
            records.extend(mode_records[key])
        return records, scanned, len(records)

    for record in tqdm(_iter_jsonl(input_path), desc=desc):
        scanned += 1
        key = _answer_acc_key(record)
        if key not in mode_records:
            continue
        mode_records[key].append(record)

    usable_ratios = {key: ratios.get(key, 0.0) for key in mode_keys}
    if any(usable_ratios.values()):
        total_max = min(
            (len(mode_records[key]) / usable_ratios[key])
            for key in mode_keys
            if usable_ratios[key] > 0
        )
        max_samples = int(total_max)
    else:
        max_samples = 0
    targets = _compute_ratio_targets(max_samples, usable_ratios, mode_keys)

    records = []
    for key in mode_keys:
        records.extend(mode_records[key][: targets.get(key, 0)])
    return records, scanned, len(records)


def _count_modes(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "wh0_woh0": 0,
        "wh0_woh1": 0,
        "wh1_woh0": 0,
        "wh1_woh1": 0,
        "missing": 0,
    }
    for record in records:
        key = _answer_acc_key(record)
        if key is None:
            counts["missing"] += 1
            continue
        if key == (0, 0):
            counts["wh0_woh0"] += 1
        elif key == (0, 1):
            counts["wh0_woh1"] += 1
        elif key == (1, 0):
            counts["wh1_woh0"] += 1
        elif key == (1, 1):
            counts["wh1_woh1"] += 1
        else:
            counts["missing"] += 1
    return counts


def collate_ospd_v3_5(
    laion_input_path: str,
    vflan_input_path: str,
    output_path: str,
    val_output_path: str,
    max_samples: int,
    answer_acc_mode: Optional[str],
    summary_json_path: Optional[str],
) -> None:
    if summary_json_path is None:
        summary_json_path = os.path.join(os.path.dirname(output_path), "summary.json")
    jsonl_path = os.path.join(os.path.dirname(output_path), "all.jsonl")

    per_source_max = max_samples // 2 if max_samples else 0
    acc_mode = _parse_answer_acc_mode(answer_acc_mode)
    balanced_mode = answer_acc_mode == "balanced"
    combined_optimal1_mode = answer_acc_mode == "combined_optimal1"

    if balanced_mode:
        laion_records, laion_scanned, laion_kept = _collect_records_balanced(
            laion_input_path, per_source_max, "Collate laion"
        )
        vflan_records, vflan_scanned, vflan_kept = _collect_records_balanced(
            vflan_input_path, per_source_max, "Collate vflan"
        )
    elif combined_optimal1_mode:
        laion_records, laion_scanned, laion_kept = _collect_records_weighted(
            laion_input_path,
            per_source_max,
            _COMBINED_OPTIMAL1_WEIGHTS,
            "Collate laion",
        )
        vflan_records, vflan_scanned, vflan_kept = _collect_records_weighted(
            vflan_input_path,
            per_source_max,
            _COMBINED_OPTIMAL1_WEIGHTS,
            "Collate vflan",
        )
    else:
        laion_records, laion_scanned, laion_kept = _collect_records(
            laion_input_path, per_source_max, acc_mode, "Collate laion"
        )
        vflan_records, vflan_scanned, vflan_kept = _collect_records(
            vflan_input_path, per_source_max, acc_mode, "Collate vflan"
        )

    records = laion_records + vflan_records

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_handle:
        for record in records:
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    mode_counts = _count_modes(records)
    if not records:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame([]).to_parquet(output_path, index=False)
        os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
        pd.DataFrame([]).to_parquet(val_output_path, index=False)
        if summary_json_path:
            os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
            with open(summary_json_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "laion_input_path": laion_input_path,
                        "vflan_input_path": vflan_input_path,
                        "output_path": output_path,
                        "val_output_path": val_output_path,
                        "total_records": 0,
                        "train_records": 0,
                        "val_records": 0,
                        "max_samples": max_samples,
                        "per_source_max": per_source_max,
                        "scanned_rows": laion_scanned + vflan_scanned,
                        "kept_rows": 0,
                        "jsonl_output_path": jsonl_path,
                        "answer_acc_mode": answer_acc_mode,
                        "kept_laion": laion_kept,
                        "kept_vflan": vflan_kept,
                        "mode_counts": mode_counts,
                    },
                    handle,
                    indent=4,
                    ensure_ascii=False,
                )
        print("[done] no valid records")
        return

    val_size = max(1, int(round(len(records) * 0.02)))
    split_idx = max(0, len(records) - val_size)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(train_records).to_parquet(output_path, index=False)
    os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
    pd.DataFrame(val_records).to_parquet(val_output_path, index=False)
    if summary_json_path:
        os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
        with open(summary_json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "laion_input_path": laion_input_path,
                    "vflan_input_path": vflan_input_path,
                    "output_path": output_path,
                    "val_output_path": val_output_path,
                    "total_records": len(records),
                    "train_records": len(train_records),
                    "val_records": len(val_records),
                    "max_samples": max_samples,
                    "per_source_max": per_source_max,
                    "scanned_rows": laion_scanned + vflan_scanned,
                    "kept_rows": len(records),
                    "jsonl_output_path": jsonl_path,
                    "answer_acc_mode": answer_acc_mode,
                    "kept_laion": laion_kept,
                    "kept_vflan": vflan_kept,
                    "mode_counts": mode_counts,
                },
                handle,
                indent=4,
                ensure_ascii=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate OPSD v3.5 data without generation."
    )
    parser.add_argument(
        "--laion-input",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.5/laion/all.jsonl",
        help="Input laion jsonl path.",
    )
    parser.add_argument(
        "--vflan-input",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.5/vflan/all.jsonl",
        help="Input vflan jsonl path.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.5",
        help="Output root directory.",
    )
    parser.add_argument(
        "--train-parquet-name",
        default="train.parquet",
        help="Train parquet name.",
    )
    parser.add_argument(
        "--val-parquet-name",
        default="validation.parquet",
        help="Validation parquet name.",
    )
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples (0 = all).")
    parser.add_argument(
        "--answer-acc-mode",
        default=None,
        help="Filter mode: wh1_woh0, balanced, or combined_optimal1 (optional).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for dataset summary json.",
    )
    args = parser.parse_args()

    output_dir = args.output_root
    if args.answer_acc_mode:
        output_dir = os.path.join(output_dir, args.answer_acc_mode)
    output_path = os.path.join(output_dir, args.train_parquet_name)
    val_output_path = os.path.join(output_dir, args.val_parquet_name)
    collate_ospd_v3_5(
        laion_input_path=args.laion_input,
        vflan_input_path=args.vflan_input,
        output_path=output_path,
        val_output_path=val_output_path,
        max_samples=args.max_samples,
        answer_acc_mode=args.answer_acc_mode,
        summary_json_path=args.summary_json,
    )


if __name__ == "__main__":
    main()


'''

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_ospd_v3.5.py \
    --answer-acc-mode combined_optimal1
    --answer-acc-mode balanced
'''