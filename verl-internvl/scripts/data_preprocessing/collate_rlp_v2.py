import argparse
import base64
import json
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

_BEDROCK_CLIENTS: Dict[str, Any] = {}
_BEDROCK_REGIONS = ("eu-north-1", "eu-west-1", "eu-west-2", "eu-west-3")
_BEDROCK_RR_IDX = 0
_CLAUDE_MODEL_ID = "eu.anthropic.claude-opus-4-5-20251101-v1:0"
_CLAUDE_SYSTEM = "You are a helpful assistant that follows instructions precisely."
_CLAUDE_THINKING_BUDGET = 1024


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _next_region() -> str:
    global _BEDROCK_RR_IDX
    region = _BEDROCK_REGIONS[_BEDROCK_RR_IDX % len(_BEDROCK_REGIONS)]
    _BEDROCK_RR_IDX += 1
    return region


def _load_bedrock(region: str):
    if region not in _BEDROCK_CLIENTS:
        _BEDROCK_CLIENTS[region] = boto3.client(
            service_name="bedrock-runtime", region_name=region
        )
    return _BEDROCK_CLIENTS[region]


def _encode_image(image_path: str) -> Tuple[str, str]:
    with open(image_path, "rb") as handle:
        image_bytes = handle.read()
    ext = os.path.splitext(image_path)[1].lower()
    media_type = "image/png" if ext in {".png"} else "image/jpeg"
    return base64.b64encode(image_bytes).decode("utf-8"), media_type


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _select_with_claude(caption: str, image_path: str) -> Optional[Dict[str, Any]]:
    if not caption or not image_path:
        return None
    image_base64, media_type = _encode_image(image_path)
    prompt = (
        "Given the image and the ground-truth caption, select ONE word that is suitable "
        "for prediction. Suitable means a visually grounded word whose value can be inferred "
        "from the image (objects, attributes like color/size/material, or relations/positions). "
        "Avoid function words, verbs, and abstract terms.\n\n"
        "Return JSON ONLY with keys: selected_word, pos_tag, masked_caption, candidate_words.\n"
        "pos_tag must be one of: noun, adjective, number, relation.\n"
        "- masked_caption must contain [Mask] exactly once.\n"
        "- Replacing [Mask] with selected_word must reproduce the caption exactly.\n"
        "- candidate_words is a list of alternative valid words (strings), exclude selected_word.\n"
        "- If pos_tag is noun, include singular/plural variants of selected_word when applicable.\n"
        "- candidate_words must not hallucinate: only include words that also fit the caption and image.\n"
        "Example output (format only):\n"
        '{"selected_word":"<word>","pos_tag":"<noun|adjective|number|relation>",'
        '"masked_caption":"<caption with [Mask]>","candidate_words":["<alt1>","<alt2>"]}\n\n'
        f"Caption: {caption}"
    )
    body = json.dumps(
        {
            "max_tokens": 3000,
            "anthropic_version": "bedrock-2023-05-31",
            "system": _CLAUDE_SYSTEM,
            "thinking": {"type": "enabled", "budget_tokens": _CLAUDE_THINKING_BUDGET},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                    ],
                }
            ],
        }
    )
    response = _load_bedrock(_next_region()).invoke_model(body=body, modelId=_CLAUDE_MODEL_ID)
    response_body = json.loads(response.get("body").read())
    text = "".join(
        part.get("text", "")
        for part in response_body.get("content", [])
        if part.get("type") == "text"
    ).strip()
    data = _extract_json(text)
    if not data:
        return None
    selected_word = str(data.get("selected_word", "")).strip()
    pos_tag = str(data.get("pos_tag", "")).strip()
    masked_caption = str(data.get("masked_caption", "")).strip()
    candidate_words = data.get("candidate_words") or []
    if not selected_word or "[Mask]" not in masked_caption:
        return None
    if masked_caption.replace("[Mask]", selected_word) != caption:
        return None
    if not isinstance(candidate_words, list):
        candidate_words = []
    return {
        "selected_word": selected_word,
        "pos_tag": pos_tag,
        "masked_caption": masked_caption,
        "candidate_words": [str(word).strip() for word in candidate_words if str(word).strip()],
    }

def _process_row(
    index: int,
    row: Dict,
    mask_token: str,
    data_source: str,
    split: str,
    image_root: str,
) -> Optional[Dict[str, Any]]:
    caption = row["conversations"][1]["value"]
    image_rel = row.get("image")
    if not image_rel:
        return None
    image_path = os.path.join(image_root, image_rel)
    if not os.path.exists(image_path):
        return None
    selection = _select_with_claude(caption, image_path)
    if not selection:
        return None
    masked_caption = selection["masked_caption"]
    masked_caption_prompt = masked_caption.replace("[Mask]", mask_token)
    question = (
        f"<image>\nCaption: {masked_caption_prompt} \n"
        "In the caption above, there is ONE missing word marked as "
        f"{mask_token} and it is a {selection['pos_tag']}. Please find it out. "
    )
    prompt = [
        {
            "content": (
                "You are an AI assistant that rigorously follows this response "
                "protocol:\n\n1. First, conduct a detailed analysis of the "
                "question. Consider different angles, potential solutions, and "
                "reason through the problem step-by-step. Enclose this entire "
                "thinking process within <think>*</think> tags.\n\n2. After "
                "the thinking section, provide a clear, concise, and direct answer "
                r"to the user's question within \boxed{}."
                '\n\n'
                r"Output format: <think>[thinking process]</think>\boxed{[answer]}"
            ),
            "role": "system",
        },
        {
            "content": question,
            "role": "user",
        },
    ]
    return {
        "data_source": data_source,
        "prompt": prompt,
        "images": [image_path],
        "ability": "caption",
        # "reward_model": {"ground_truth": selection["selected_word"], "style": "rule"},
        "reward_model": {"ground_truth": [selection["selected_word"]] + selection["candidate_words"], "style": "rule"},
        "extra_info": {
            "answer": selection["selected_word"],
            "index": index,
            "pos_tag": selection["pos_tag"],
            "masked_caption": masked_caption,
            "candidate_words": selection["candidate_words"],
            "question": question,
            "split": split,
            "image_paths": [image_path],
        },
    }


def collate_missing_word_jsonl(
    input_path: str,
    output_path: str,
    val_output_path: str,
    seed: int = 13,
    mask_token: str = "[MASK]",
    data_source: str = "coco_karpathy_50k",
    split: str = "train",
    max_samples: int = 50000,
    image_root: str = "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco",
    summary_json_path: Optional[str] = None,
    num_workers: int = 8,
    collate_sample: bool = False,
) -> None:
    _ = seed
    if summary_json_path is None:
        # base, _ext = os.path.splitext(output_path)
        # summary_json_path = f"{base}.summary.json"
        summary_json_path = os.path.join(os.path.dirname(output_path), "summary.json")
    records: List[Dict] = []
    jsonl_path = os.path.join(os.path.dirname(output_path), "all.jsonl")
    if collate_sample:
        records = list(_iter_jsonl(jsonl_path))
        if max_samples and len(records) > max_samples:
            records = records[:max_samples]
        stats = {"scanned": len(records), "kept": len(records)}
    else:
        rows = list(_iter_jsonl(input_path))
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        write_lock = threading.Lock()
        scanned = 0
        jsonl_handle = open(jsonl_path, "w", encoding="utf-8")
        try:
            with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
                futures = [
                    executor.submit(
                        _process_row, i, row, mask_token, data_source, split, image_root
                    )
                    for i, row in enumerate(rows)
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
                    scanned += 1
                    payload = future.result()
                    if payload:
                        records.append(payload)
                        with write_lock:
                            jsonl_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            jsonl_handle.flush()
                        if len(records) % 50 == 0:
                            print(f"[thread] collected {len(records)} samples")
                        if max_samples and len(records) >= max_samples:
                            for pending in futures:
                                pending.cancel()
                            break
        finally:
            jsonl_handle.close()
        stats = {"scanned": scanned, "kept": len(records)}
    summary_meta = {
        "data_source": data_source,
        "split": split,
        "mask_token": mask_token,
        "image_root": image_root,
        "jsonl_output_path": jsonl_path,
        "selector": "claude_bedrock",
        "bedrock_model_id": _CLAUDE_MODEL_ID,
        "bedrock_regions": list(_BEDROCK_REGIONS),
        "thinking": {"enabled": True, "budget_tokens": _CLAUDE_THINKING_BUDGET},
        "num_workers": num_workers,
    }
    if max_samples and len(records) > max_samples:
        records = records[:max_samples]
        stats["kept"] = min(stats["kept"], max_samples)
    if not records:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(records).to_parquet(output_path, index=False)
        os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
        pd.DataFrame(records).to_parquet(val_output_path, index=False)
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
                        "scanned_rows": stats["scanned"],
                        "kept_rows": stats["kept"],
                        **summary_meta,
                    },
                    handle,
                    indent=4,
                    ensure_ascii=False,
                )
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
                    "input_path": input_path,
                    "output_path": output_path,
                    "val_output_path": val_output_path,
                    "total_records": len(records),
                    "train_records": len(train_records),
                    "val_records": len(val_records),
                    "max_samples": max_samples,
                    "scanned_rows": stats["scanned"],
                    "kept_rows": stats["kept"],
                    **summary_meta,
                },
                handle,
                indent=4,
                ensure_ascii=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create missing-word prompts from COCO captions."
    )
    parser.add_argument("--input", required=True, help="Input jsonl path.")
    parser.add_argument(
        "--output",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v2/train.parquet",
        help="Train parquet path.",
    )
    parser.add_argument(
        "--val-output",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v2/validation.parquet",
        help="Validation parquet path.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples.")
    parser.add_argument("--mask-token", default="[MASK]", help="Mask token.")
    parser.add_argument(
        "--data-source", default="coco_karpathy_50k", help="Data source label."
    )
    parser.add_argument("--split", default="train", help="Split label.")
    parser.add_argument("--num-workers", type=int, default=8, help="Thread workers.")
    parser.add_argument(
        "--collate-sample",
        action="store_true",
        help="Collate train/val/summary from existing all.jsonl.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for dataset summary json.",
    )
    parser.add_argument(
        "--image-root",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco",
        help="COCO image root for resolving relative paths.",
    )
    args = parser.parse_args()
    collate_missing_word_jsonl(
        input_path=args.input,
        output_path=args.output,
        val_output_path=args.val_output,
        seed=args.seed,
        mask_token=args.mask_token,
        data_source=args.data_source,
        split=args.split,
        max_samples=args.max_samples,
        image_root=args.image_root,
        summary_json_path=args.summary_json,
        num_workers=args.num_workers,
        collate_sample=args.collate_sample,
    )


if __name__ == "__main__":
    main()



'''
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_rlp_v2.py \
    --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco/annotations/coco_karpathy_train_50k.jsonl \
    --output /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v2/train.parquet \
    --val-output /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v2/validation.parquet \
    --summary-json /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v2/summary.json \
    --num-workers 40 \
    --collate-sample \
    --max_samples 50 
'''