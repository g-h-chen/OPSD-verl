import argparse
import base64
import io
import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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


def _build_user_prompt(question: str, hint: str, include_image_token: bool = True) -> str:
    hint_block = f"Image description:\n{hint}\n\n" if hint else ""
    image_token = "<image>\n" if include_image_token else ""
    return f"{image_token}{hint_block}{question.replace('<image>', '')}\n"


def _build_messages(question: str, hint: str) -> List[Dict[str, str]]:
    user_prompt = _build_user_prompt(question, hint)
    return [
        {"content": _SYSTEM_PROMPT, "role": "system"},
        {"content": user_prompt, "role": "user"},
    ]


def _is_valid(record: Dict[str, Any]) -> bool:
    image_path = record.get("image")
    captions = record.get("captions") or []
    question = record.get("question")
    answers = record.get("answers") or []
    return bool(image_path and captions and question and answers)


def _select_caption(captions: Sequence[str]) -> str:
    return captions[0]
    # for caption in captions:
    #     caption = str(caption).strip()
    #     if caption:
    #         return caption
    # return ""


def _parse_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"<think>.*?</think>.*?\\boxed\{(.+?)\}\s*$", text, re.S)
    if not match:
        return None
    return match.group(1).strip()


def _normalize_answer(text: str) -> str:
    normalized = str(text).strip().lower()
    if normalized.startswith("$") and normalized.endswith("$"):
        normalized = normalized[1:-1].strip()
    return normalized


def _is_correct(pred: Optional[str], answers: Sequence[str]) -> bool:
    if pred is None:
        return False
    pred_norm = _normalize_answer(pred)
    for answer in answers:
        if pred_norm == _normalize_answer(answer):
            return True
    return False


def uuid_str_from_str(s: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    return str(uuid.uuid5(namespace, s))


def _build_record(
    index: int,
    record: Dict[str, Any],
    hint: str,
    data_source: str,
    split: str,
    answer_w_hint: Optional[str] = None,
    answer_wo_hint: Optional[str] = None,
    w_h_c: Optional[bool] = None,
    wo_h_c: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    image_path = record.get("image", "")
    captions = record.get("captions") or []
    question = str(record.get("question", "")).strip()
    answers = record.get("answers") or []
    if not (image_path and captions and question and answers):
        return None
    prompt = _build_messages(question, hint)
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
            "answer_w_hint": answer_w_hint,
            "answer_wo_hint": answer_wo_hint,
            "w_h_c": w_h_c,
            "wo_h_c": wo_h_c,
        },
    }


def _build_prompt_text(tokenizer: AutoTokenizer, question: str, hint: str) -> str:
    messages = _build_messages(question, hint)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _encode_image_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _call_vllm_api(
    api_base: str,
    api_key: Optional[str],
    api_model: str,
    question: str,
    hint: str,
    image_b64: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: float,
) -> str:
    user_text = _build_user_prompt(question, hint, include_image_token=False)
    payload: Dict[str, Any] = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k >= 0:
        payload["top_k"] = top_k
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        f"{api_base}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def collate_ospd_v3_5_laion(
    input_path: str,
    output_path: str,
    val_output_path: str,
    max_samples: int,
    data_source: str,
    split: str,
    summary_json_path: Optional[str],
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    batch_size: int,
    shard_count: int,
    shard_id: int,
    use_vllm_api: bool,
    vllm_api_base: str,
    vllm_api_model: str,
    vllm_api_key: Optional[str],
    vllm_api_timeout: float,
    vllm_api_workers: int,
) -> None:
    if summary_json_path is None:
        summary_json_path = os.path.join(os.path.dirname(output_path), "summary.json")
    jsonl_path = os.path.join(os.path.dirname(output_path), "all.jsonl")

    tokenizer = None
    sampling_params = None
    llm = None
    if not use_vllm_api:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    records: List[Dict[str, Any]] = []
    scanned = 0
    kept = 0
    kept_a_correct = 0
    kept_b_correct = 0
    shard_scanned = 0
    pending: List[Dict[str, Any]] = []

    def _flush_pending() -> None:
        nonlocal kept, kept_a_correct, kept_b_correct, pending
        if not pending:
            return
        outputs: List[str] = []
        if use_vllm_api:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max(1, vllm_api_workers)) as executor:
                futures = {}
                for idx, item in enumerate(pending):
                    image_b64 = _encode_image_base64(item["image"])
                    futures[
                        executor.submit(
                            _call_vllm_api,
                            vllm_api_base,
                            vllm_api_key,
                            vllm_api_model,
                            item["question"],
                            item["caption"],
                            image_b64,
                            max_new_tokens,
                            temperature,
                            top_p,
                            top_k,
                            vllm_api_timeout,
                        )
                    ] = (idx, 0)
                    futures[
                        executor.submit(
                            _call_vllm_api,
                            vllm_api_base,
                            vllm_api_key,
                            vllm_api_model,
                            item["question"],
                            "",
                            image_b64,
                            max_new_tokens,
                            temperature,
                            top_p,
                            top_k,
                            vllm_api_timeout,
                        )
                    ] = (idx, 1)
                output_pairs = [["", ""] for _ in pending]
                for future in as_completed(futures):
                    idx, which = futures[future]
                    output_pairs[idx][which] = future.result()
            for output_pair in output_pairs:
                outputs.extend(output_pair)
        else:
            assert tokenizer is not None and sampling_params is not None and llm is not None
            prompts: List[Dict[str, Any]] = []
            for item in pending:
                prompt_a = _build_prompt_text(tokenizer, item["question"], item["caption"])
                prompt_b = _build_prompt_text(tokenizer, item["question"], "")
                prompts.append({"prompt": prompt_a, "multi_modal_data": {"image": item["image"]}})
                prompts.append({"prompt": prompt_b, "multi_modal_data": {"image": item["image"]}})
            outputs_raw = llm.generate(prompts, sampling_params)
            if len(outputs_raw) != len(prompts):
                pending = []
                return
            for output in outputs_raw:
                outputs.append(output.outputs[0].text if output.outputs else "")
        for i, item in enumerate(pending):
            output_a = outputs[2 * i] if 2 * i < len(outputs) else ""
            output_b = outputs[2 * i + 1] if 2 * i + 1 < len(outputs) else ""
            answer_a = _parse_boxed_answer(output_a)
            answer_b = _parse_boxed_answer(output_b)
            is_a_correct = _is_correct(answer_a, item["answers"])
            is_b_correct = _is_correct(answer_b, item["answers"])
            if is_a_correct:
                kept_a_correct += 1
            if is_b_correct:
                kept_b_correct += 1
            payload = _build_record(
                kept,
                item["record"],
                "",
                data_source,
                split,
                answer_w_hint=answer_a,
                answer_wo_hint=answer_b,
                w_h_c=is_a_correct,
                wo_h_c=is_b_correct,
            )
            # print(f"answer_a: {answer_a}, answer_b: {answer_b}")
            if not payload:
                continue
            records.append(payload)
            jsonl_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 50 == 0:
                print(f"[kept] {kept} samples")
        pending = []
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_handle:
        overall_bar = tqdm(_iter_jsonl(input_path), desc="Overall")
        for record in overall_bar:
            scanned += 1
            if max_samples and kept >= max_samples:
                break
            if not _is_valid(record):
                continue
            shard_scanned += 1
            if shard_count > 1 and (shard_scanned - 1) % shard_count != shard_id:
                continue

            image_path = record.get("image", "")
            captions = record.get("captions") or []
            question = str(record.get("question", "")).strip()
            answers = record.get("answers") or []
            caption = _select_caption(captions)
            if not caption:
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            pending.append(
                {
                    "record": record,
                    "image": image,
                    "question": question,
                    "caption": caption,
                    "answers": answers,
                }
            )
            if len(pending) >= max(1, batch_size):
                _flush_pending()

        _flush_pending()

    if summary_json_path:
        os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
        with open(summary_json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "input_path": input_path,
                    "total_records": len(records),
                    "max_samples": max_samples,
                    "scanned_rows": scanned,
                    "kept_rows": kept,
                    "jsonl_output_path": jsonl_path,
                    "data_source": data_source,
                    "split": split,
                    "with_hint_correct": kept_a_correct,
                    "without_hint_correct": kept_b_correct,
                    "model_path": model_path,
                    "use_vllm_api": use_vllm_api,
                    "vllm_api_base": vllm_api_base,
                    "vllm_api_model": vllm_api_model,
                    "batch_size": batch_size,
                    "shard_count": shard_count,
                    "shard_id": shard_id,
                },
                handle,
                indent=4,
                ensure_ascii=False,
            )
    if not records:
        print("[done] no valid records")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate OPSD v3.5 data with vLLM rejection sampling."
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
        "--output-root",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3.5/laion",
        help="Output root directory for summary/jsonl (no parquet written).",
    )
    parser.add_argument(
        "--train-parquet-name",
        default="train.parquet",
        help="Train parquet name (used to derive output directory only).",
    )
    parser.add_argument(
        "--val-parquet-name",
        default="validation.parquet",
        help="Validation parquet name (used to derive output directory only).",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples (0 = all).")
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
    parser.add_argument(
        "--model-path",
        default="/home/efs/hardychen/models/InternVL3_5-4B-Pretrained",
        help="VLLM model path for rejection sampling.",
    )
    parser.add_argument(
        "--use-vllm-api",
        action="store_true",
        help="Use vLLM OpenAI API server instead of local vLLM.",
    )
    parser.add_argument(
        "--vllm-api-base",
        default="http://localhost:8000",
        help="vLLM OpenAI API base URL.",
    )
    parser.add_argument(
        "--vllm-api-model",
        default="",
        help="Model name for vLLM API (defaults to --model-path).",
    )
    parser.add_argument(
        "--vllm-api-key",
        default=None,
        help="Optional API key for vLLM server.",
    )
    parser.add_argument(
        "--vllm-api-timeout",
        type=float,
        default=120.0,
        help="Timeout for vLLM API requests (seconds).",
    )
    parser.add_argument(
        "--vllm-api-workers",
        type=int,
        default=16,
        help="Number of concurrent API requests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=2048, help="Number of records per vLLM batch.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of shards for DP.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard id for this process.")
    args = parser.parse_args()
    output_path = os.path.join(args.output_root, args.train_parquet_name)
    val_output_path = os.path.join(args.output_root, args.val_parquet_name)
    vllm_api_model = args.vllm_api_model or args.model_path
    vllm_api_base = args.vllm_api_base.rstrip("/")
    collate_ospd_v3_5_laion(
        input_path=args.input,
        output_path=output_path,
        val_output_path=val_output_path,
        max_samples=args.max_samples,
        data_source=args.data_source,
        split=args.split,
        summary_json_path=args.summary_json,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        shard_count=args.shard_count,
        shard_id=args.shard_id,
        use_vllm_api=args.use_vllm_api,
        vllm_api_base=vllm_api_base,
        vllm_api_model=vllm_api_model,
        vllm_api_key=args.vllm_api_key,
        vllm_api_timeout=args.vllm_api_timeout,
        vllm_api_workers=args.vllm_api_workers,
    )


if __name__ == "__main__":
    main()

