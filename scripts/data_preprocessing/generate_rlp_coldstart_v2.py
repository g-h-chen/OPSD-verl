import argparse
import base64
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
from tqdm import tqdm

_BEDROCK_CLIENTS: Dict[str, Any] = {}
_BEDROCK_REGIONS = ("eu-north-1", "eu-west-1", "eu-west-2", "eu-west-3")
_BEDROCK_RR_IDX = 0
_CLAUDE_MODEL_ID = "eu.anthropic.claude-opus-4-5-20251101-v1:0"
_CLAUDE_SYSTEM = "You are a helpful assistant that follows instructions precisely."
_CLAUDE_THINKING_BUDGET = 1024

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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


def _normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = text.strip().strip("`\"' ")
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    text = text.strip(" .,:;!?()[]{}")
    return text


def _extract_answer(text: str) -> Tuple[bool, Optional[str]]:
    think_match = _THINK_RE.search(text)
    box_match = _BOX_RE.search(text)
    if not box_match:
        return bool(think_match), None
    answer = box_match.group(1).strip()
    return bool(think_match), answer


def _resolve_image_path(image_path: str, image_root: Optional[str]) -> str:
    if image_root and image_path and not os.path.isabs(image_path):
        return os.path.join(image_root, image_path)
    return image_path


def _relativize_image_path(image_path: str, image_root: Optional[str]) -> str:
    if image_root and image_path and image_path.startswith(image_root):
        return os.path.relpath(image_path, image_root)
    return image_path


def _prompt_to_messages(
    prompt: List[Dict[str, Any]],
    image_path: Optional[str],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    system_text = None
    messages: List[Dict[str, Any]] = []
    image_payload = None
    if image_path:
        image_base64, media_type = _encode_image(image_path)
        image_payload = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_base64,
            },
        }
    for item in prompt:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role == "system":
            system_text = content
            continue
        if role == "user":
            content_list = [{"type": "text", "text": content}]
            if image_payload is not None:
                content_list.append(image_payload)
            messages.append({"role": "user", "content": content_list})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
    return system_text, messages


def _invoke_claude(
    prompt: List[Dict[str, Any]],
    image_path: Optional[str],
    max_tokens: int,
    max_retries: int,
    temperature: float,
) -> Optional[str]:
    system_text, messages = _prompt_to_messages(prompt, image_path)
    if not messages:
        return None
    body = {
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "thinking": {"type": "enabled", "budget_tokens": _CLAUDE_THINKING_BUDGET},
        "temperature": temperature,
    }
    body["system"] = system_text if system_text is not None else ""
    last_error = None
    for _ in range(max_retries):
        try:
            response = _load_bedrock(_next_region()).invoke_model(
                body=json.dumps(body),
                modelId=_CLAUDE_MODEL_ID,
            )
            response_body = json.loads(response.get("body").read())
            text = "".join(
                part.get("text", "")
                for part in response_body.get("content", [])
                if part.get("type") == "text"
            ).strip()
            if text:
                return text
        except Exception as exc:
            last_error = exc
    if last_error:
        print(f"[warn] Claude request failed: {last_error}")
    return None


def _process_record(
    index: int,
    record: Dict[str, Any],
    image_root: Optional[str],
    max_tokens: int,
    max_retries: int,
    temperature: float,
    keep_invalid: bool,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "missing_image": 0,
        "no_response": 0,
        "invalid_format": 0,
        "incorrect_answer": 0,
        "kept": 0,
    }
    prompt = record.get("prompt") or []
    images = record.get("images") or []
    image_path = _resolve_image_path(images[0], image_root) if images else None
    if not image_path or not os.path.exists(image_path):
        stats["missing_image"] += 1
        return None, stats
    response = _invoke_claude(
        prompt=prompt,
        image_path=image_path,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
    )
    if not response:
        stats["no_response"] += 1
        return None, stats
    has_think, extracted = _extract_answer(response)
    if not has_think or not extracted:
        print(f"[warn] Invalid format: {response}")
        stats["invalid_format"] += 1
    ground_truth = record.get("reward_model", {}).get("ground_truth", [])
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    normalized_gt = {_normalize_answer(str(item)) for item in ground_truth if str(item).strip()}
    normalized_answer = _normalize_answer(extracted) if extracted else ""
    is_correct = bool(normalized_answer and normalized_answer in normalized_gt)
    if not is_correct:
        stats["incorrect_answer"] += 1
    if not (keep_invalid or (has_think and extracted and is_correct)):
        return None, stats

    conversations = []
    for item in prompt:
        role = item.get("role")
        if role == "system":
            conversations.append({"from": "system", "value": item.get("content", "")})
        elif role == "user":
            conversations.append({"from": "human", "value": item.get("content", "")})
    conversations.append({"from": "gpt", "value": response})

    payload = {
        "image": _relativize_image_path(image_path, image_root),
        "conversations": conversations,
        "data_source": record.get("data_source"),
        "reward_model": record.get("reward_model"),
        "extra_info": record.get("extra_info"),
        "response": response,
        "extracted_answer": extracted,
        "format_ok": bool(has_think and extracted),
        "is_correct": is_correct,
        "source_index": index,
    }
    stats["kept"] += 1
    return payload, stats


def generate_coldstart_jsonl(
    input_path: str,
    output_path: str,
    image_root: Optional[str],
    max_samples: int,
    num_workers: int,
    max_tokens: int,
    max_retries: int,
    temperature: float,
    keep_invalid: bool,
) -> None:
    records = list(_iter_jsonl(input_path))
    if max_samples and len(records) > max_samples:
        records = records[:max_samples]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_lock = threading.Lock()
    stats = {
        "scanned": 0,
        "kept": 0,
        "missing_image": 0,
        "no_response": 0,
        "invalid_format": 0,
        "incorrect_answer": 0,
    }
    with open(output_path, "w", encoding="utf-8") as out_handle:
        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [
                executor.submit(
                    _process_record,
                    idx,
                    record,
                    image_root,
                    max_tokens,
                    max_retries,
                    temperature,
                    keep_invalid,
                )
                for idx, record in enumerate(records)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generate"):
                stats["scanned"] += 1
                payload, local_stats = future.result()
                for key, val in local_stats.items():
                    stats[key] += val
                if payload:
                    with write_lock:
                        out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        out_handle.flush()
                if stats["scanned"] % 50 == 0:
                    print(
                        f"[progress] scanned={stats['scanned']} kept={stats['kept']} "
                        f"missing_image={stats['missing_image']} invalid_format={stats['invalid_format']} "
                        f"incorrect_answer={stats['incorrect_answer']} no_response={stats['no_response']}"
                    )
    print("[done] stats:", json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate coldstart traces for RLP v2 via Claude."
    )
    parser.add_argument(
        "--input",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v2/all.jsonl",
        help="Input collated jsonl path.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/"
            "internvl_chat_gpt_oss/data/rlp_coldstart/v2/"
            "rlp_voldstart_v2_generation.jsonl"
        ),
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--image-root",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco",
        help="Image root for relativizing paths.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all).")
    parser.add_argument("--num-workers", type=int, default=20, help="Thread workers.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for Claude.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per request.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep responses even if format/answer is invalid.",
    )
    args = parser.parse_args()

    generate_coldstart_jsonl(
        input_path=args.input,
        output_path=args.output,
        image_root=args.image_root,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        temperature=args.temperature,
        keep_invalid=args.keep_invalid,
    )


if __name__ == "__main__":
    main()

