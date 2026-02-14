import argparse
import json
import os
from typing import Any, Dict, Iterable, List

from tqdm import tqdm


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
            conversations = record.get("conversations") or []
            if not conversations or conversations[0].get("from") != "system":
                conversations = [{"from": "system", "value": ""}] + conversations
            payload = {
                "image": image_path,
                "width": 0,
                "height": 0,
                "conversations": conversations,
                "length": _estimate_text_tokens(conversations),
                "image_count": 1,
            }
            out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total += 1
    print(f"[done] collated {total} samples -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate RLP coldstart generations into InternVL jsonl."
    )
    parser.add_argument(
        "--input",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/"
            "internvl_chat_gpt_oss/data/rlp_coldstart/v2/"
            "rlp_voldstart_v2_generation.jsonl"
        ),
        help="Generated coldstart jsonl path.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/"
            "internvl_chat_gpt_oss/data/rlp_coldstart/v2/"
            "rlp_coldstart_v2_collated.jsonl"
        ),
        help="Output collated jsonl path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
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

