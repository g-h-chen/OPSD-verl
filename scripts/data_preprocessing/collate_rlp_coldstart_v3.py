import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm
from PIL import Image

_SYSTEM_PROMPT = (
    "You are an AI assistant that rigorously follows this response "
    "protocol:\n\n1. First, conduct a detailed analysis of the "
    "question. Consider different angles, potential solutions, and "
    "reason through the problem step-by-step. Enclose this entire "
    "thinking process within <think>*</think> tags.\n\n2. After "
    "the thinking section, provide a clear, concise, and direct answer "
    r"to the user's question within \boxed{}."
    + "\n\n"
    + r"Output format: <think>[thinking process]</think>\boxed{[answer]}"
)


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


def _build_user_prompt(caption: str, question: str) -> str:
    caption = caption.strip()
    question = question.strip()
    # return f"<image>\nImage description:\n{caption}\n\n{question}"
    return f"<image>\n{question}"


def _get_image_size(image_path: str) -> Tuple[int, int]:
    if not image_path or not os.path.exists(image_path):
        return 0, 0
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return int(width), int(height)
    except Exception:
        return 0, 0


def collate_coldstart_jsonl(
    input_path: str,
    output_path: str,
    max_samples: int,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as out_handle:
        for record in tqdm(_iter_jsonl(input_path), desc="Collate"):
            if max_samples and total >= max_samples:
                break
            image_path = record.get("image", "")
            captions = record.get("captions") or []
            question = record.get("question", "")
            thinking = record.get("thinking", "")
            answers = record.get("answers") or []
            if not (image_path and captions and question and thinking and answers):
                continue
            caption0 = str(captions[0]) if captions else ""
            user_prompt = _build_user_prompt(caption0, str(question))
            assistant_answer = f"<think>{thinking}</think>\\boxed{{{answers[0]}}}"
            width, height = _get_image_size(image_path)
            conversations = [
                {"from": "system", "value": _SYSTEM_PROMPT},
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
    print(f"[done] collated {total} samples -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate RLP v3 generations into InternVL jsonl for SFT."
    )
    parser.add_argument(
        "--input",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v3/"
            # "allava_laion_caption_375k_rlp_v3_generation.jsonl"
            "allava_vflan_caption_195k_rlp_v3_generation.jsonl"
        ),
        help="Generated v3 jsonl path.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/"
            "internvl_chat_gpt_oss/data/rlp_coldstart/v3/"
            # "rlp_coldstart_v3_collated.jsonl"
            "rlp_coldstart_v3_allava_vflan_caption_collated.jsonl"
        ),
        help="Output collated jsonl path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples to collate (0 = all).",
    )
    args = parser.parse_args()
    collate_coldstart_jsonl(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
