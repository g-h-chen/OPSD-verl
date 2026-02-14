import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

_SYSTEM_PROMPT = (
    "You are an AI assistant that rigorously follows this response "
    "protocol:\n\n1. First, conduct a detailed analysis of the "
    "question. Consider different angles, potential solutions, and "
    "reason through the problem step-by-step. Enclose this entire "
    "thinking process within <think>*</think> tags.\n\n2. After "
    "the thinking section, provide a clear, concise, and direct answer "
    r"to the user's question within \boxed{}."
    "\n\n"
    r"Output format: <think>[thinking process]</think>\boxed{[answer]}"
)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_skip_image_paths(path: str) -> set:
    if not path or not os.path.exists(path):
        return set()
    skip_paths = set()
    for record in _iter_jsonl(path):
        image_path = str(record.get("image", "")).strip()
        if image_path:
            skip_paths.add(image_path)
    return skip_paths


def _build_user_prompt(question: str, hint: str) -> str:
    hint_block = f"Image description:\n{hint}\n\n" if hint else ""
    return f"<image>\n{hint_block}{question}\n"


def _normalize_probs(weights: Sequence[float]) -> List[float]:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return [0.0 for _ in weights]
    return [max(0.0, w) / total for w in weights]


def _interp_probs(
    ratio: float, anchors: Sequence[Tuple[float, Sequence[float]]]
) -> List[float]:
    if ratio <= anchors[0][0]:
        return list(anchors[0][1])
    if ratio >= anchors[-1][0]:
        return list(anchors[-1][1])
    for (r0, p0), (r1, p1) in zip(anchors[:-1], anchors[1:]):
        if r0 <= ratio <= r1:
            t = (ratio - r0) / max(1e-8, r1 - r0)
            return [
                (1.0 - t) * float(a) + t * float(b)
                for a, b in zip(p0, p1)
            ]
    return list(anchors[-1][1])


def _sample_from_probs(rng: random.Random, probs: Sequence[float]) -> int:
    roll = rng.random()
    cumulative = 0.0
    for idx, p in enumerate(probs):
        cumulative += p
        if roll <= cumulative:
            return idx
    return max(0, len(probs) - 1)


# def _select_hint(
#     captions: List[str],
#     index: int,
#     total: int,
#     rng: random.Random,
# ) -> str:
#     if not captions:
#         return ""
#     if total <= 1:
#         return captions[0]
#     ratio = index / max(1, total - 1)
#     anchors = [
#         (0.00, (0.88, 0.10, 0.02, 0.00)),
#         (0.11, (0.78, 0.17, 0.04, 0.01)),
#         (0.22, (0.68, 0.22, 0.08, 0.02)),
#         (0.33, (0.55, 0.30, 0.10, 0.05)),
#         (0.44, (0.42, 0.32, 0.18, 0.08)),
#         (0.55, (0.30, 0.32, 0.24, 0.14)),
#         (0.66, (0.20, 0.35, 0.25, 0.20)),
#         (0.77, (0.12, 0.25, 0.25, 0.38)),
#         (0.88, (0.07, 0.15, 0.22, 0.56)),
#         (1.00, (0.05, 0.10, 0.20, 0.65)),
#     ]
#     probs = _normalize_probs(_interp_probs(ratio, anchors))
#     choice = _sample_from_probs(rng, probs)
#     if choice == 0:
#         return captions[0] if len(captions) > 0 else ""
#     if choice == 1:
#         return captions[1] if len(captions) > 1 else captions[0]
#     if choice == 2:
#         return captions[2] if len(captions) > 2 else captions[-1]
#     return ""


def _is_valid(record: Dict[str, Any]) -> bool:
    image_path = record.get("image")
    captions = record.get("captions") or []
    question = record.get("question")
    answers = record.get("answers") or []
    return bool(image_path and captions and question and answers)


def _build_record(
    index: int,
    record: Dict[str, Any],
    hint: str,
    data_source: str,
    split: str,
) -> Optional[Dict[str, Any]]:
    image_path = record.get("image", "")
    captions = record.get("captions") or []
    question = str(record.get("question", "")).strip()
    answers = record.get("answers") or []
    if not (image_path and captions and question and answers):
        return None
    user_prompt = _build_user_prompt(question, hint)
    prompt = [
        {"content": _SYSTEM_PROMPT, "role": "system"},
        {"content": user_prompt, "role": "user"},
    ]
    return {
        "data_source": data_source,
        "prompt": prompt,
        "images": [image_path],
        "ability": "caption",
        "reward_model": {"ground_truth": answers, "style": "rule"},
        "extra_info": {
            "index": index,
            "uid": uuid_str_from_str(f"{data_source}_{split}_{index}"),
            "question": question,
            "captions": captions,
            "answer": answers,
            "split": split,
            "image_paths": [image_path],
            "hint": hint,
        },
    }

import uuid

def uuid_str_from_str(s: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    return str(uuid.uuid5(namespace, s))
    
def collate_rlp_v3(
    input_path: str,
    output_path: str,
    val_output_path: str,
    max_samples: int,
    data_source: str,
    split: str,
    summary_json_path: Optional[str],
    seed: int,
    coldstart_jsonl_path: Optional[str],
) -> None:
    if summary_json_path is None:
        summary_json_path = os.path.join(os.path.dirname(output_path), "summary.json")
    jsonl_path = os.path.join(os.path.dirname(output_path), "all.jsonl")

    skip_image_paths = _load_skip_image_paths(coldstart_jsonl_path)
    print(f"Skipping {len(skip_image_paths)} coldstart images")
    total_valid = 0
    scanned = 0
    skipped_coldstart = 0
    for record in _iter_jsonl(input_path):
        scanned += 1
        if not _is_valid(record):
            continue
        if record.get("image") in skip_image_paths:
            skipped_coldstart += 1
            continue
        total_valid += 1
        if max_samples and total_valid >= max_samples:
            break

    if total_valid == 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame([]).to_parquet(output_path, index=False)
        os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
        pd.DataFrame([]).to_parquet(val_output_path, index=False)
        if summary_json_path:
            os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
            with open(summary_json_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "input_path": input_path,
                        "output_path": output_path,
                        "val_output_path": val_output_path,
                        "total_records": 0,
                        "train_records": 0,
                        "val_records": 0,
                        "max_samples": max_samples,
                        "scanned_rows": scanned,
                        "kept_rows": 0,
                        "skipped_coldstart_rows": skipped_coldstart,
                        "jsonl_output_path": jsonl_path,
                        "data_source": data_source,
                        "split": split,
                        "hint_schedule": "quartiles: c0 -> c1 -> c2 -> none",
                    },
                    handle,
                    indent=4,
                    ensure_ascii=False,
                )
        print("[done] no valid records")
        return

    records: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    kept = 0
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_handle:
        for record in tqdm(_iter_jsonl(input_path), desc="Collate"):
            if max_samples and kept >= max_samples:
                break
            if not _is_valid(record):
                continue
            if record.get("image") in skip_image_paths:
                continue
            # hint = _select_hint(record.get("captions") or [], kept, total_valid, rng)
            # v3.1: no hint, cuz we train this into the sft warmup
            hint = ""
            payload = _build_record(kept, record, hint, data_source, split)
            if not payload:
                continue
            records.append(payload)
            jsonl_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1

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
                    "input_path": input_path,
                    "output_path": output_path,
                    "val_output_path": val_output_path,
                    "total_records": len(records),
                    "train_records": len(train_records),
                    "val_records": len(val_records),
                    "max_samples": max_samples,
                    "scanned_rows": scanned,
                    "kept_rows": kept,
                    "skipped_coldstart_rows": skipped_coldstart,
                    "jsonl_output_path": jsonl_path,
                    "data_source": data_source,
                    "split": split,
                        "hint_schedule": "probabilistic interpolation: c0->c1->c2->none",
                        "hint_seed": seed,
                    "hint_schedule": "probabilistic interpolation: c0->c1->c2->none",
                    "hint_seed": seed,
                },
                handle,
                indent=4,
                ensure_ascii=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate RLP v3.1 generations into train/val parquet."
    )
    parser.add_argument(
        "--input",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3/"
            "allava_laion_caption_375k_rlp_v3_generation.jsonl"
        ),
        help="Input RLP v3 jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.1-exclude_coldstart/train.parquet",
        help="Train parquet path.",
    )
    parser.add_argument(
        "--val-output",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.1-exclude_coldstart/validation.parquet",
        help="Validation parquet path.",
    )
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples (0 = all).")
    parser.add_argument(
        "--data-source",
        default="allava_laion_caption",
        help="Data source label.",
    )
    parser.add_argument("--split", default="train", help="Split label.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for dataset summary json.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for hint sampling.")
    parser.add_argument(
        "--coldstart-jsonl",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/"
            "internvl_chat_gpt_oss/data/rlp_coldstart/v3.1/rlp_coldstart_v3.1_collated.jsonl"
        ),
        help="Coldstart collated jsonl path for exclusion (by image_path).",
    )
    args = parser.parse_args()
    collate_rlp_v3(
        input_path=args.input,
        output_path=args.output,
        val_output_path=args.val_output,
        max_samples=args.max_samples,
        data_source=args.data_source,
        split=args.split,
        summary_json_path=args.summary_json,
        seed=args.seed,
        coldstart_jsonl_path=args.coldstart_jsonl,
    )


if __name__ == "__main__":
    main()