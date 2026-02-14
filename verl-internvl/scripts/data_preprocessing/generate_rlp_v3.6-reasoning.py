import argparse
import base64
import json
import os
import random
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
_JUDGE_MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
_CLAUDE_SYSTEM = "You are a helpful assistant that follows instructions precisely."
_CLAUDE_THINKING_BUDGET = 2000
_VLAA_SFT_PATH = (
    "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/"
    "VLAA-Thinking/VLAA-Thinking-SFT-126K.json"
)
_VLAA_GRPO_PATH = (
    "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/"
    "VLAA-Thinking/VLAA-Thinking-GRPO-25K.json"
)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_json_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        first = None
        for line in handle:
            if line.strip():
                first = line
                break
        if first is None:
            return
        stripped = first.lstrip()
        if stripped.startswith("["):
            payload = stripped + handle.read()
            data = json.loads(payload)
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        yield json.loads(first)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_image_files(image_dir: str) -> Iterable[str]:
    supported_ext = {".jpg", ".jpeg", ".png"}
    for root, _, files in os.walk(image_dir):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in supported_ext:
                yield os.path.join(root, name)


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


def _infer_image_id(image_path: str) -> str:
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)
    return stem


def _build_dataset_source(dataset: str) -> str:
    dataset = dataset.strip().strip("/")
    if dataset.startswith("vlaa_thinking-"):
        return dataset
    return f"vlaa_thinking-{dataset}"


def _dataset_name_from_source(dataset_source: str) -> str:
    if dataset_source.startswith("vlaa_thinking-"):
        return dataset_source[len("vlaa_thinking-") :]
    return dataset_source


def _resolve_output_path(
    output_path: Optional[str],
    output_dir: str,
    dataset_source: str,
) -> str:
    if output_path:
        return output_path
    filename = f"{dataset_source}-reasoning_generation.jsonl"
    return os.path.join(output_dir, filename)


def _extract_gt_from_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    boxed = re.search(r"\\boxed\{([^}]*)\}", text, flags=re.DOTALL)
    if boxed:
        return boxed.group(1).strip()
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer:
        return answer.group(1).strip()
    return None


def _normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.replace("\x08", "")
    match = re.search(r"\\boxed\{([^}]*)\}", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_judge_result(text: str) -> Optional[bool]:
    # try:
    #     payload = json.loads(text)
    #     if isinstance(payload, dict) and isinstance(payload.get("correct"), bool):
    #         return payload["correct"]
    # except json.JSONDecodeError:
    #     pass
    # lowered = text.strip().lower()
    # if lowered in {"true", "false"}:
    #     return lowered == "true"
    if text.strip() == '1':
        return True
    return False
    # return None


def _judge_answer(question: str, gt: str, image_path: str, model_answer: str, max_retries: int) -> Optional[bool]:
    boxed = _extract_boxed_answer(model_answer)
    instructions = (
        "You are a strict judge. Compare the model's answer to the ground truth.\n"
        "Note that the questions may be shown in the image, so you should judge comprehensively.\n"
        "Return a binary result ONLY: 1 for correct, 0 for incorrect. NOTHING ELSE.\n"
        "**Textual question**: {question}\n\n"
        "**Ground truth**: {gt}\n\n"
        # "**Model boxed answer**: {boxed}\n\n"
        "**Model full answer**: {model_answer}\n\n"
    ).format(
        question=question,
        gt=gt,
        boxed=boxed or "",
        model_answer=model_answer,
    )
    prompt = [
        {"role": "system", "content": _CLAUDE_SYSTEM},
        {"role": "user", "content": instructions},
    ]
    response = _invoke_model(
        model_id=_JUDGE_MODEL_ID,
        prompt=prompt,
        image_path=image_path,
        max_tokens=5,
        max_retries=max_retries,
        temperature=0.0,
        thinking=False,
    )
    # if response == '0':
    #     breakpoint()
    if not response:
        return None
    return _extract_judge_result(response)


def _build_vlaa_question_mapping(dataset_name: str) -> Dict[str, List[Dict[str, str]]]:
    prefix = f"{dataset_name}/"
    mapping: Dict[str, List[Dict[str, str]]] = {}
    for path in (_VLAA_SFT_PATH, _VLAA_GRPO_PATH):
        for record in _iter_json_records(path):
            image_rel = record.get("image")
            if not isinstance(image_rel, str) or not image_rel.startswith(prefix):
                continue
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            gt = record.get("gt")
            if not isinstance(gt, str) or not gt.strip():
                gt = _extract_gt_from_answer(record.get("answer"))
            if not gt:
                continue
            mapping.setdefault(image_rel, []).append(
                {"question": question.strip(), "gt": str(gt).strip()}
            )
    return mapping


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


def _invoke_model(
    model_id: str,
    prompt: List[Dict[str, Any]],
    image_path: Optional[str],
    max_tokens: int,
    max_retries: int,
    temperature: float,
    thinking: bool = True,
) -> Optional[str]:
    system_text, messages = _prompt_to_messages(prompt, image_path)
    if not messages:
        return None
    body = {
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "thinking": {"type": "enabled", "budget_tokens": _CLAUDE_THINKING_BUDGET} if thinking else {"type": "disabled"},
        "temperature": temperature,
    }
    body["system"] = system_text if system_text is not None else ""
    last_error = None
    for _ in range(max_retries):
        try:
            response = _load_bedrock(_next_region()).invoke_model(
                body=json.dumps(body),
                modelId=model_id,
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


def _invoke_claude(
    prompt: List[Dict[str, Any]],
    image_path: Optional[str],
    max_tokens: int,
    max_retries: int,
    temperature: float,
) -> Optional[str]:
    return _invoke_model(
        model_id=_CLAUDE_MODEL_ID,
        prompt=prompt,
        image_path=image_path,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
    )


def _build_prompt() -> List[Dict[str, Any]]:
    instructions = (
        "You are given an image. Follow the steps and return JSON only.\n"
        "Step 1: Generate ONE reasoning QA pair about the image.\n"
        "- The question must be answerable directly from the image without external knowledge.\n"
        "- Make it a reasoning-style question that requires multiple visual cues.\n"
        "- Prefer a short answer (AT MOST TWO words). If not possible, use multiple-choice with 4 options.\n"
        "- If multiple-choice, include options in the question as \"A) ... B) ... C) ... D) ...\".\n"
        "- Produce a list of candidate answers (strings), covering simple variants.\n"
        "  Examples: for a letter answer use [\"A\", \"Option A\"]; "
        "for a number use [\"11\", \"eleven\"]; for a short phrase consider potential variants (case ignored).\n"
        "- Provide a detailed reasoning process that references only the image.\n"
        "- The model_answer field must follow this format exactly: "
        "<think>[thinking process]</think>\\boxed{[answer]}\n"
        "Step 2: Generate ONE hint that helps answer the question.\n"
        "- The hint should guide a model toward the relevant visual cues without revealing the answer.\n"
        "- Keep it short and grounded in the image.\n"
        "Critical constraints:\n"
        "- Do NOT use markdown or extra text.\n"
        "Output JSON format:\n"
        "{"
        "\"question\": \"...\", "
        "\"model_answer\": \"...\", "
        "\"hint\": \"...\", "
        "\"answers\": [\"...\"]"
        "}\n"
        "Return only valid JSON."
    )
    return [
        {"role": "system", "content": _CLAUDE_SYSTEM},
        {"role": "user", "content": instructions},
    ]


        # "- The hint should guide a model toward the relevant visual cues without revealing the answer.\n"
        # "- Keep it short and grounded in the image.\n"
def _build_prompt_with_question(question: str) -> List[Dict[str, Any]]:

    instructions = (
        "You are given an image and a question. Follow the steps and return JSON only.\n"
        f"Question: {question}\n"
        "Step 1: Answer the question about the image.\n"
        "- The answer must be directly supported by the image without external knowledge.\n"
        "- Prefer a short answer (AT MOST TWO words). If not possible, use multiple-choice with 4 options.\n"
        "- If multiple-choice, include options in the question as \"A) ... B) ... C) ... D) ...\".\n"
        "- Produce a list of candidate answers (strings), covering simple variants.\n"
        "  Examples: for a letter answer use [\"A\", \"Option A\"]; "
        "for a number use [\"11\", \"eleven\"]; for a short phrase consider potential variants (case ignored).\n"
        "- Provide a detailed reasoning process that references only the image.\n"
        "- The model_answer field must follow this format exactly: "
        "<think>[thinking process]</think>\\boxed{[answer]}\n"
        "Step 2: Generate ONE hint that helps answer the question.\n"
        "- The hint should provide sufficient information to answer the question (e.g. a description of relevant information in the image).\n"
        "- Ideally, an incapable model should be able to answer the question with the hint.\n"
        "Critical constraints:\n"
        "- Do NOT use markdown or extra text.\n"
        "Output JSON format:\n"
        "{"
        "\"question\": \"...\", "
        "\"model_answer\": \"...\", "
        "\"hint\": \"...\", "
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
    for key in ("question", "model_answer", "hint"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    answers = payload.get("answers")
    if not isinstance(answers, list) or not answers:
        return False
    if not all(isinstance(item, str) and item.strip() for item in answers):
        return False
    return True


def _has_think_boxed_format(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.replace("\x08", r"\b").strip()
    if not stripped.startswith("<think>"):
        return False
    pattern = r"<think>.*</think>\s*\\boxed\{.*\}\s*$"
    return re.search(pattern, stripped, flags=re.DOTALL) is not None


def get_synthesis_question() -> str:
    pool = [
        # "Find the function expression in the image.",
        # 'What is the function expression in the image?',
        # "What is the function equation in the image?",
        # 'Find the function equation in the image.',
        'Find out the function equation in the image.',
        'Solve the function equation in the image.',
    ]
    return random.choice(pool)


def _process_record(
    index: int,
    record: Dict[str, Any],
    image_root: Optional[str],
    max_tokens: int,
    max_retries: int,
    format_retries: int,
    answer_retries: int,
    use_original_question: bool,
    temperature: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (output, outcome) where outcome is one of:
    missing_image, no_response, invalid_json, invalid_fields,
    invalid_format, incorrect_answer, kept."""
    image_path = record.get("image")
    if not image_path:
        return None, "missing_image"
    image_path = _resolve_image_path(image_path, image_root)
    if not image_path or not os.path.exists(image_path):
        return None, "missing_image"
    payload = None
    response_text = None
    last_failure = "no_response"
    question = record.get("question") if use_original_question else None
    if record['meta']['dataset_source'] == 'vlaa_thinking-synthesis':
        # normalize question
        if 'gen_func' in image_path:
            # if question
            question = get_synthesis_question()
            # + '\n' + question.split('\n')[1].strip()

    if use_original_question and not question:
        msg = (
            f"[error] --use-original-question is set but no question found for "
            f"record index={index}, image={image_path}"
        )
        print(msg)
        raise ValueError(msg)

    gt = record.get("gt") if use_original_question else None
    prompt = _build_prompt_with_question(question) if use_original_question else _build_prompt()
    for _ in range(max(1, answer_retries)):
        response_text = None
        for _ in range(max(1, format_retries)):
            response_text = _invoke_claude(
                prompt=prompt,
                image_path=image_path,
                max_tokens=max_tokens,
                max_retries=max_retries,
                temperature=temperature,
            )
            if not response_text:
                last_failure = "no_response"
                continue
            payload = _extract_json(response_text)
            if not payload:
                print(f"[warn] Invalid JSON response: {response_text}")
                last_failure = "invalid_json"
                payload = None
                continue
            if not _validate_payload(payload):
                print(f"[warn] Missing or invalid fields: {response_text}")
                last_failure = "invalid_fields"
                payload = None
                continue
            if not _has_think_boxed_format(payload.get("model_answer", "")):
                print(f"[warn] Invalid thinking format: {response_text}")
                last_failure = "invalid_format"
                payload = None
                continue
            break
        if not response_text:
            continue
        if payload is None:
            continue
        if use_original_question and gt:
            judged = _judge_answer(
                question=question or payload.get("question", ""),
                image_path=image_path,
                gt=gt,
                model_answer=payload.get("model_answer", ""),
                max_retries=max_retries,
            )
            # breakpoint()
            if judged is not True:
                last_failure = "incorrect_answer"
                payload = None
                continue
        break
    if not response_text:
        return None, last_failure
    if payload is None:
        return None, last_failure

    output_question = question if use_original_question and question else payload["question"]
    output_meta = dict(record.get("meta") or {})
    if gt:
        output_meta["gt"] = gt

    meta = output_meta
    meta.update(
        {
            "image_id": _infer_image_id(image_path),
            "question": output_question,
            "answers": payload["answers"],
            "model_answer": payload["model_answer"],
            "hint": payload["hint"],
            "model_id": _CLAUDE_MODEL_ID,
        }
    )
    output = {
        "image": image_path,
        "question": output_question,
        "model_answer": payload["model_answer"],
        "hint": payload["hint"],
        "answers": payload["answers"],
        "meta": meta,
    }
    return output, "kept"


def generate_rlp_v3_jsonl(
    input_path: str,
    output_path: Optional[str],
    output_dir: str,
    image_root: Optional[str],
    source: Optional[str],
    dataset: Optional[str],
    use_original_question: bool,
    max_samples: int,
    num_workers: int,
    max_tokens: int,
    max_retries: int,
    format_retries: int,
    answer_retries: int,
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
        if use_original_question and dataset_source.startswith("vlaa_thinking-"):
            question_mapping = _build_vlaa_question_mapping(dataset_name)
            if not question_mapping:
                msg = (
                    f"[error] --use-original-question is set but no questions found "
                    f"for dataset '{dataset_name}' in VLAA-Thinking files"
                )
                print(msg)
                raise ValueError(msg)
            records = []
            for path in image_files:
                rel_path = os.path.relpath(path, input_path)
                entries = question_mapping.get(rel_path, [])
                if not entries:
                    msg = (
                        f"[error] --use-original-question is set but no question "
                        f"found for image: {rel_path}"
                    )
                    print(msg)
                    raise ValueError(msg)
                for entry in entries:
                    records.append(
                        {
                            "image": os.path.abspath(path),
                            "question": entry["question"],
                            "gt": entry["gt"],
                            "meta": {
                                "source": resolved_source,
                                "dataset_source": dataset_source,
                            },
                        }
                    )
        else:
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
    use_original_questions = (
        use_original_question
        and os.path.isdir(input_path)
        and dataset_source.startswith("vlaa_thinking-")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_lock = threading.Lock()
    kept_images: set = set()  # track images that already have a kept trace
    stats = {
        "scanned": 0,
        "kept": 0,
        "missing_image": 0,
        "no_response": 0,
        "invalid_json": 0,
        "invalid_fields": 0,
        "invalid_format": 0,
        "incorrect_answer": 0,
        "duplicate_image": 0,
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
                    format_retries,
                    answer_retries,
                    use_original_questions,
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
                        img_key = payload["image"]
                        if img_key in kept_images:
                            stats["duplicate_image"] += 1
                            stats["kept"] -= 1  # undo the kept count from _process_record
                        else:
                            kept_images.add(img_key)
                            out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            out_handle.flush()
                if stats["scanned"] % 50 == 0:
                    print(
                        f"[progress] scanned={stats['scanned']} kept={stats['kept']} "
                        f"missing_image={stats['missing_image']} invalid_json={stats['invalid_json']} "
                        f"invalid_fields={stats['invalid_fields']} invalid_format={stats['invalid_format']} "
                        f"incorrect_answer={stats['incorrect_answer']} "
                        f"no_response={stats['no_response']} "
                        f"duplicate_image={stats['duplicate_image']}"
                    )
    print("[done] stats:", json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate image reasoning QA (RLP v3 reasoning) via Claude."
    )
    parser.add_argument(
        "--input",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco/train2014",
        help="Input jsonl path or image directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output jsonl path. If omitted, uses {dataset_source}-reasoning_generation.jsonl.",
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
    parser.add_argument(
        "--use-original-question",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use original VLAA-Thinking questions when available.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all).")
    parser.add_argument("--num-workers", type=int, default=20, help="Thread workers.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for Claude.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per request.")
    parser.add_argument(
        "--format-retries",
        type=int,
        default=2,
        help="Retries for invalid output format.",
    )
    parser.add_argument(
        "--answer-retries",
        type=int,
        default=3,
        help="Retries when generated answer does not match GT.",
    )
    parser.add_argument("--temperature", type=float, default=1., help="Sampling temperature.")
    args = parser.parse_args()

    generate_rlp_v3_jsonl(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.output_dir,
        image_root=args.image_root,
        source=args.source,
        dataset=args.dataset,
        use_original_question=args.use_original_question,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        format_retries=args.format_retries,
        answer_retries=args.answer_retries,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()


'''
# Reasoning generation examples
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco/train2014 \
  --dataset coco_train2014 \
  --source coco/train2014

# running aws3-5
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-synthesis \
  --source VLAA-Thinking \
--use-original-question

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-arxivqa \
  --source VLAA-Thinking \
--use-original-question

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-vizwiz \
  --source VLAA-Thinking \
--use-original-question



python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-chartqa \
  --source VLAA-Thinking \
--use-original-question


python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-clevr_math \
  --source VLAA-Thinking \
--use-original-question


python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-docvqa \
  --source VLAA-Thinking \
--use-original-question

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-geoqa170k \
  --source VLAA-Thinking

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/generate_rlp_v3.6-reasoning.py \
  --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/VLAA-Thinking/images \
  --dataset vlaa_thinking-vizwiz \
  --source VLAA-Thinking
'''