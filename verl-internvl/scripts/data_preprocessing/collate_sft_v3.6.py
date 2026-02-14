import argparse
import json
import os
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

# Output format follows:
# /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_rlp_coldstart_v3.1.py


_PACIFIC = timezone(timedelta(hours=-8), name="US/Pacific")


def _pacific_timestamp() -> str:
    """Return the current time as an ISO-8601 string in US/Pacific."""
    return datetime.now(_PACIFIC).strftime("%Y-%m-%d %H:%M:%S %Z")


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


def _estimate_text_tokens(conversations: List[Dict[str, Any]]) -> int:
    text = "\n".join(str(turn.get("value", "")) for turn in conversations)
    token_count = len(text.split()) * 3
    return max(1, token_count)


def _get_image_size(image_path: str) -> Tuple[int, int]:
    if not image_path or not os.path.exists(image_path):
        return 0, 0
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return int(width), int(height)
    except Exception:
        return 0, 0


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


def _collect_records_multi(
    input_paths: Sequence[str],
    input_names: Sequence[str],
    max_samples: int,
    answer_acc_mode: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    dataset_order: List[str] = []
    if not input_paths:
        return records, dataset_order
    per_input_max = 0
    if max_samples:
        per_input_max = max_samples // len(input_paths)
        remainder = max_samples % len(input_paths)
    else:
        remainder = 0
    acc_mode = _parse_answer_acc_mode(answer_acc_mode)
    balanced_mode = answer_acc_mode == "balanced"
    combined_optimal1_mode = answer_acc_mode == "combined_optimal1"
    for idx, input_path in enumerate(input_paths):
        name = input_names[idx] if idx < len(input_names) else os.path.basename(input_path)
        local_max = 0
        if max_samples:
            local_max = per_input_max + (1 if idx < remainder else 0)
        desc = f"Collate {os.path.basename(os.path.dirname(input_path))}"
        if balanced_mode:
            recs, _, _ = _collect_records_balanced(input_path, local_max, desc)
        elif combined_optimal1_mode:
            recs, _, _ = _collect_records_weighted(
                input_path, local_max, _COMBINED_OPTIMAL1_WEIGHTS, desc
            )
        else:
            recs, _, _ = _collect_records(input_path, local_max, acc_mode, desc)
        records.extend(recs)
        dataset_order.extend([name] * len(recs))
    return records, dataset_order


def _extract_prompt(prompt: Any, role: str) -> str:
    if not isinstance(prompt, list):
        return ""
    for turn in prompt:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("role", "")).lower() == role:
            return str(turn.get("content", ""))
    return ""


def _extract_first_image(images: Any) -> str:
    if isinstance(images, list) and images:
        return str(images[0])
    if isinstance(images, str):
        return images
    return ""


def _to_internvl_relpath(path: str) -> str:
    """
    Convert an absolute path under InternVL/internvl_chat_gpt_oss to a repo-relative path,
    e.g. ".../InternVL/internvl_chat_gpt_oss/data/xxx" -> "data/xxx".
    Fallback: return original path.
    """
    marker = "/InternVL/internvl_chat_gpt_oss/"
    idx = path.find(marker)
    if idx < 0:
        return path
    return path[idx + len(marker) :]


def _load_vlaa_model_answers(
    dataset_name: str,
    reasoning: bool,
    v3_generation_root: str,
) -> Dict[str, str]:
    """
    Load {image_path: model_answer} from:
      vlaa_thinking-{ds_name}{-reasoning}_generation.jsonl
    """
    prefix = "vlaa_thinking-"
    if not dataset_name.startswith(prefix):
        raise ValueError(
            f"Only vlaa_thinking-* datasets are supported for SFT answer retrieval; got: {dataset_name}"
        )
    ds_name = dataset_name[len(prefix) :]
    suffix = "-reasoning" if reasoning else ""
    gen_path = os.path.join(
        v3_generation_root, f"vlaa_thinking-{ds_name}{suffix}_generation.jsonl"
    )
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"Missing generation jsonl for assistant_answer: {gen_path}")
    answers: Dict[str, str] = {}
    for rec in tqdm(_iter_jsonl(gen_path), desc=f"Load answers {os.path.basename(gen_path)}"):
        image_path = str(rec.get("image", ""))
        model_answer = rec.get("model_answer", None)
        if not image_path:
            continue
        if model_answer is None:
            continue
        model_answer = str(model_answer)
        if image_path in answers and answers[image_path] != model_answer:
            print(f'got duplicate model_answer')
            continue
            # raise ValueError(
            #     "Conflicting model_answer for the same image path:\n"
            #     f"- image: {image_path}\n"
            #     f"- a: {answers[image_path][:200]}\n"
            #     f"- b: {model_answer[:200]}"
            # )
        answers[image_path] = model_answer
    return answers


def collate_sft_v3_6(
    input_paths: Sequence[str],
    input_names: Sequence[str],
    output_jsonl_path: str,
    info_json_path: str,
    info_dataset_key: str,
    max_samples: int,
    answer_acc_mode: Optional[str],
    repeat_to_balance: bool,
    shuffle: bool,
    seed: int,
    reasoning: bool,
    v3_generation_root: str,
    retrieve_from_opsd: bool,
    opsd_input_path: Optional[str],
) -> None:
    records: List[Dict[str, Any]] = []
    dataset_order: List[str] = []
    if not retrieve_from_opsd:
        records, dataset_order = _collect_records_multi(
            input_paths=input_paths,
            input_names=input_names,
            max_samples=max_samples,
            answer_acc_mode=answer_acc_mode,
        )

        repeat_target = max_samples if max_samples else 0
        if repeat_to_balance and repeat_target and records:
            per_input_max = repeat_target // len(input_names)
            remainder = repeat_target % len(input_names)
            per_dataset_targets = {
                name: per_input_max + (1 if idx < remainder else 0)
                for idx, name in enumerate(input_names)
            }
            per_dataset_records: Dict[str, List[Dict[str, Any]]] = {name: [] for name in input_names}
            for name, record in zip(dataset_order, records):
                per_dataset_records[name].append(record)
            repeated_records: List[Dict[str, Any]] = []
            for name in input_names:
                items = per_dataset_records.get(name, [])
                target = per_dataset_targets.get(name, 0)
                if not items or not target:
                    continue
                if len(items) < target:
                    repeats = target // len(items)
                    extra = target % len(items)
                    items = items * repeats + items[:extra]
                repeated_records.extend(items)
            records = repeated_records

        if shuffle and records:
            rng = random.Random(seed)
            rng.shuffle(records)

    # Load assistant answers once (join key = absolute image path).
    answers_by_image: Dict[str, str] = {}
    for name in input_names:
        local = _load_vlaa_model_answers(
            dataset_name=name,
            reasoning=reasoning,
            v3_generation_root=v3_generation_root,
        )
        for image_path, ans in local.items():
            if image_path in answers_by_image and answers_by_image[image_path] != ans:
                raise ValueError(f"Conflicting model_answer across datasets for image: {image_path}")
            answers_by_image[image_path] = ans

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    total = 0
    with open(output_jsonl_path, "w", encoding="utf-8") as out_handle:
        if retrieve_from_opsd:
            if not opsd_input_path:
                raise ValueError("opsd_input_path must be provided when retrieve_from_opsd=True")
            if not os.path.exists(opsd_input_path):
                raise FileNotFoundError(f"Missing OPSD input jsonl: {opsd_input_path}")
            iterator = _iter_jsonl(opsd_input_path)
        else:
            iterator = records
        for record in tqdm(iterator, desc="Write SFT"):
            if max_samples and total >= max_samples:
                break
            image_path = _extract_first_image(record.get("images") or record.get("image"))
            if not image_path:
                raise ValueError(f"Missing image path in record: keys={list(record.keys())}")
            if image_path not in answers_by_image:
                raise KeyError(f"assistant_answer not found for image: {image_path}")
            system_prompt = _extract_prompt(record.get("prompt"), "system")
            user_prompt = _extract_prompt(record.get("prompt"), "user")
            if not system_prompt or not user_prompt:
                raise ValueError(
                    f"Missing system/user prompt for image={image_path}. "
                    f"prompt_keys={record.get('prompt')}"
                )
            assistant_answer = answers_by_image[image_path]
            width, height = _get_image_size(image_path)
            conversations = [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_answer},
            ]
            payload = {
                "image": image_path,
                "width": width,
                "height": height,
                "conversations": conversations,
                "length": _estimate_text_tokens(conversations),
                "image_count": 1,
            }
            out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total += 1
    print(f"[done] collated {total} samples -> {output_jsonl_path}")

    # Write dataset registry (info.json), matching keys from rlp_coldstart_v3.1.json + timestamp.
    info_payload = {
        info_dataset_key: {
            "root": "",
            "annotation": _to_internvl_relpath(output_jsonl_path),
            "data_augment": False,
            "repeat_time": 1,
            "length": total,
            "timestamp": _pacific_timestamp(),
        }
    }
    os.makedirs(os.path.dirname(info_json_path), exist_ok=True)
    with open(info_json_path, "w", encoding="utf-8") as info_handle:
        json.dump(info_payload, info_handle, indent=4, ensure_ascii=False)
    print(f"[done] wrote info -> {info_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collate OPSD v3.6 records into InternVL SFT jsonl.")
    parser.add_argument(
        "--input-root",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.6",
        help="Root directory for per-dataset all.jsonl folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset folder names to concatenate (e.g., vlaa_thinking-geoqa170k).",
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use *_reasoning dataset folders if set (and join against *-reasoning_generation.jsonl).",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/"
            "internvl_chat_gpt_oss/data/collated/v3.6"
        ),
        help="Output root directory.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples (0 = all).")
    parser.add_argument(
        "--repeat-to-balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Repeat samples to reach max-samples when under target.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle records after repeat-to-balance.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--answer-acc-mode",
        default=None,
        help="Filter mode: wh1_woh0, balanced, or combined_optimal1 (optional).",
    )
    parser.add_argument(
        "--v3-generation-root",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3",
        help="Root directory containing v3 vlaa_thinking-*_generation.jsonl files.",
    )
    parser.add_argument(
        "--retrieve-from-opsd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, read images from OPSD-collated jsonl at "
            "verl-internvl/datasets/rlp/v3.6/{ds_short_names}/{ans_acc_mode}/all.jsonl "
            "and follow that file order strictly. "
            "When enabled, do not use --shuffle/--repeat-to-balance."
        ),
    )
    args = parser.parse_args()

    if not args.datasets:
        raise ValueError("Provide datasets via --datasets.")
    if args.retrieve_from_opsd and (args.shuffle or args.repeat_to_balance):
        raise ValueError("Do not set --shuffle/--repeat-to-balance when --retrieve-from-opsd is enabled.")

    input_names: List[str] = []
    sorted_datasets = sorted(args.datasets)
    suffix = "_reasoning" if args.reasoning else ""
    input_paths: List[str] = []
    for name in sorted_datasets:
        input_names.append(name)
        input_paths.append(os.path.join(args.input_root, f"{name}{suffix}", "all.jsonl"))

    missing = [path for path in input_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing input paths: {missing}")

    display_name_map = {
        "vlaa_thinking-synthesis": "vlt-synthesis",
        "vlaa_thinking-geoqa170k": "vlt-geoqa170k",
    }
    display_names = [display_name_map.get(name, name) for name in input_names]
    ds_short_names = f"{'--'.join(sorted(display_names))}_{'reasoning' if args.reasoning else 'alignment'}"
    ans_acc_mode = args.answer_acc_mode or "all"
    output_dir = os.path.join(args.output_root, ds_short_names, ans_acc_mode)
    output_jsonl_path = os.path.join(output_dir, "all.jsonl")
    info_json_path = os.path.join(output_dir, "info.json")
    info_dataset_key = f"sft_v3.6_{ds_short_names}_{ans_acc_mode}"
    opsd_input_path: Optional[str] = None
    if args.retrieve_from_opsd:
        opsd_input_path = os.path.join(args.input_root, ds_short_names, ans_acc_mode, "all.jsonl")

    collate_sft_v3_6(
        input_paths=input_paths,
        input_names=input_names,
        output_jsonl_path=output_jsonl_path,
        info_json_path=info_json_path,
        info_dataset_key=info_dataset_key,
        max_samples=args.max_samples,
        answer_acc_mode=args.answer_acc_mode,
        repeat_to_balance=args.repeat_to_balance,
        shuffle=args.shuffle,
        seed=args.seed,
        reasoning=args.reasoning,
        v3_generation_root=args.v3_generation_root,
        retrieve_from_opsd=args.retrieve_from_opsd,
        opsd_input_path=opsd_input_path,
    )


if __name__ == "__main__":
    main()


"""
Sample commands:

# alignment (single dataset)
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_sft_v3.6.py \
  --input-root /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.6 \
  --datasets vlaa_thinking-geoqa170k \
  --max-samples 100 \
  --answer-acc-mode wh1_woh0 \
  --shuffle \
  --seed 42

# alignment (multi datasets)
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_sft_v3.6.py \
  --input-root /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.6 \
  --datasets vlaa_thinking-geoqa170k vlaa_thinking-synthesis \
  --max-samples 40000 \
  --answer-acc-mode wh1_woh0 \
  --repeat-to-balance \
  --shuffle \

# reasoning (joins against v3 *-reasoning_generation.jsonl)
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_sft_v3.6.py \
  --input-root /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.6 \
  --datasets vlaa_thinking-geoqa170k vlaa_thinking-synthesis \
  --reasoning \
  --max-samples 8000 \
  --answer-acc-mode wh1_woh0 \
    --retrieve-from-opsd \
  --repeat-to-balance \
  --shuffle

"""