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
_CLAUDE_THINKING_BUDGET = 2000


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


def _resolve_image_path(image_path: str, image_root: Optional[str]) -> str:
    if image_root and image_path and not os.path.isabs(image_path):
        return os.path.join(image_root, image_path)
    return image_path


def _iter_image_files(image_dir: str) -> Iterable[str]:
    supported_ext = {".jpg", ".jpeg", ".png"}
    for root, _, files in os.walk(image_dir):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in supported_ext:
                yield os.path.join(root, name)


def _build_dataset_source(dataset: str) -> str:
    dataset = dataset.strip().strip("/")
    if dataset.startswith("vlaa_thinking-"):
        return dataset
    return f"vlaa_thinking-{dataset}"


def _dataset_name_from_source(dataset_source: str) -> str:
    if dataset_source.startswith("vlaa_thinking-"):
        return dataset_source[len("vlaa_thinking-"):]
    return dataset_source


def _resolve_output_path(
    output_path: Optional[str],
    output_dir: str,
    dataset_source: str,
) -> str:
    if output_path:
        return output_path
    filename = f"{dataset_source}_generation.jsonl"
    return os.path.join(output_dir, filename)


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


def _build_prompt() -> List[Dict[str, Any]]:
    instructions = (
        "You are given an image. Follow the steps and return JSON only.\n"
        "Step 1: Generate three captions with decreasing granularity.\n"
        "- caption1: extremely detailed, capture as much visual detail as possible.\n"
        "- caption2: moderately detailed, summarize main objects and relations.\n"
        "- caption3: coarse-grained, short high-level summary.\n"
        "Step 2: Generate ONE verifiable QA pair about the image.\n"
        "- The question must be answerable directly from the image without external knowledge.\n"
        "- Prefer a short answer (AT MOST TWO words). If not possible, use multiple-choice with 4 options.\n"
        "- If multiple-choice, include options in the question as \"A) ... B) ... C) ... D) ...\".\n"
        "- Produce a list of candidate answers (strings), covering simple variants.\n"
        "  Examples: for a letter answer use [\"A\", \"Option A\"]; "
        "for a number use [\"11\", \"eleven\"]; for a short phrase consider potential variants (case ignored).\n"
        "- Provide a detailed reasoning process that references only the image.\n"
        "Critical constraints:\n"
        "- Do NOT mention or reference the captions in the question, reasoning, or answer.\n"
        "- Pretend you only see the image.\n"
        "- Do NOT use markdown or extra text.\n"
        "Output JSON format:\n"
        "{"
        "\"captions\": [\"caption1\", \"caption2\", \"caption3\"], "
        "\"question\": \"...\", "
        "\"thinking\": \"...\", "
        "\"answers\": [\"...\"]"
        "}\n"
        "Return only valid JSON."
    )
    return [
        {"role": "system", "content": _CLAUDE_SYSTEM},
        {"role": "user", "content": instructions},
    ]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _validate_payload(payload: Dict[str, Any]) -> bool:
    captions = payload.get("captions")
    if not isinstance(captions, list) or len(captions) != 3:
        return False
    if not all(isinstance(item, str) and item.strip() for item in captions):
        return False
    for key in ("question", "thinking"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    answers = payload.get("answers")
    if not isinstance(answers, list) or not answers:
        return False
    if not all(isinstance(item, str) and item.strip() for item in answers):
        return False
    return True


def _process_record(
    index: int,
    record: Dict[str, Any],
    image_root: Optional[str],
    max_tokens: int,
    max_retries: int,
    temperature: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (output, outcome) where outcome is one of:
    missing_image, no_response, invalid_json, invalid_fields, kept."""
    image_path = record.get("image")
    if not image_path:
        return None, "missing_image"
    image_path = _resolve_image_path(image_path, image_root)
    if not image_path or not os.path.exists(image_path):
        return None, "missing_image"
    response = _invoke_claude(
        prompt=_build_prompt(),
        image_path=image_path,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
    )
    if not response:
        return None, "no_response"
    payload = _extract_json(response)
    if not payload:
        print(f"[warn] Invalid JSON response: {response}")
        return None, "invalid_json"
    if not _validate_payload(payload):
        print(f"[warn] Missing or invalid fields: {response}")
        return None, "invalid_fields"

    output = {
        "image": image_path,
        "captions": payload["captions"],
        "question": payload["question"],
        "thinking": payload["thinking"],
        "answers": payload["answers"],
        "meta": record.get("meta", {}),
    }
    return output, "kept"


def generate_rlp_v3_jsonl(
    input_path: str,
    output_path: Optional[str],
    output_dir: str,
    image_root: Optional[str],
    source: Optional[str],
    dataset: Optional[str],
    max_samples: int,
    num_workers: int,
    max_tokens: int,
    max_retries: int,
    temperature: float,
) -> None:
    dataset_name = dataset
    if not dataset_name:
        dataset_name = os.path.basename(os.path.normpath(input_path))
    dataset_source = _build_dataset_source(dataset_name)
    output_path = _resolve_output_path(output_path, output_dir, dataset_source)

    if os.path.isdir(input_path):
        dataset_name = _dataset_name_from_source(dataset_source)
        candidate_dir = os.path.join(input_path, dataset_name)
        if not os.path.isdir(candidate_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {candidate_dir}. "
                "Ensure --input points to the parent images directory and --dataset matches a subfolder."
            )
        image_dir = candidate_dir
        resolved_source = source or os.path.basename(os.path.normpath(image_dir))
        image_files = list(_iter_image_files(image_dir))
        if max_samples and len(image_files) > max_samples:
            image_files = image_files[:max_samples]
        records = [
            {
                "image": os.path.abspath(path),
                "meta": {"source": resolved_source, "dataset_source": dataset_source},
            }
            for path in image_files
        ]
    else:
        records = list(_iter_jsonl(input_path))
        if max_samples and len(records) > max_samples:
            records = records[:max_samples]
        if source or dataset_source:
            for record in records:
                meta = record.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    record["meta"] = meta
                if source:
                    meta.setdefault("source", source)
                meta.setdefault("dataset_source", dataset_source)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_lock = threading.Lock()
    stats = {
        "scanned": 0,
        "kept": 0,
        "missing_image": 0,
        "no_response": 0,
        "invalid_json": 0,
        "invalid_fields": 0,
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
                )
                for idx, record in enumerate(records)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generate"):
                stats["scanned"] += 1
                payload, outcome = future.result()
                stats[outcome] += 1
                if payload:
                    with write_lock:
                        out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        out_handle.flush()
                if stats["scanned"] % 50 == 0:
                    print(
                        f"[progress] scanned={stats['scanned']} kept={stats['kept']} "
                        f"missing_image={stats['missing_image']} invalid_json={stats['invalid_json']} "
                        f"invalid_fields={stats['invalid_fields']} no_response={stats['no_response']}"
                    )
    print("[done] stats:", json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate image captions + verifiable QA (RLP v3) via Claude."
    )
    parser.add_argument(
        "--input",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/"
            # "internvl_chat_gpt_oss/data/collated/allava_laion_caption_375k.jsonl"
            "internvl_chat_gpt_oss/data/collated/allava_vflan_caption_195k.jsonl"
        ),
        help="Input jsonl path or image directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output jsonl path. If omitted, uses {dataset_source}_generation.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3",
        help="Output directory for auto-generated filenames.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Image root for resolving paths (jsonl input only).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source label stored in meta (optional).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name or dataset_source (vlaa_thinking-{dataset}).",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all).")
    parser.add_argument("--num-workers", type=int, default=20, help="Thread workers.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for Claude.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per request.")
    parser.add_argument("--temperature", type=float, default=1., help="Sampling temperature.")
    args = parser.parse_args()

    generate_rlp_v3_jsonl(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.output_dir,
        image_root=args.image_root,
        source=args.source,
        dataset=args.dataset,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()


'''
# JSONL input
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6.py \
  --image-root "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/ALLaVA-4V/allava_vflan"

# VLAA directory input
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-chartqa \
  --source VLAA-Thinking

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-arxivqa \
  --source VLAA-Thinking

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-docvqa \
  --source VLAA-Thinking

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-vizwiz \
  --source VLAA-Thinking


'''