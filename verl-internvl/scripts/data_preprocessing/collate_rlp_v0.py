import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

_NLP = None


def _load_spacy(model: str = "en_core_web_sm"):
    global _NLP
    if _NLP is None:
        try:
            import spacy  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is required for NLP-based word selection. "
                "Install spaCy and the 'en_core_web_sm' model."
            ) from exc
        try:
            _NLP = spacy.load(model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model}' is not available. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP


def _unique_tokens(tokens: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    seen = set()
    unique: List[Tuple[int, int, str]] = []
    for start, end, text in tokens:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        unique.append((start, end, text))
    return unique


def select_key_span_spacy(
    caption: str,
    rng: Optional[random.Random] = None,
    model: str = "en_core_web_sm",
) -> Optional[Tuple[str, int, int]]:
    if not caption:
        return None
    nlp = _load_spacy(model)
    doc = nlp(caption)
    candidates: List[Tuple[int, int, str]] = []
    for ent in doc.ents:
        token = ent.root
        if token.is_stop or token.is_punct:
            continue
        candidates.append((token.idx, token.idx + len(token.text), token.text))
    for chunk in doc.noun_chunks:
        token = chunk.root
        if token.is_stop or token.is_punct:
            continue
        candidates.append((token.idx, token.idx + len(token.text), token.text))
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        if token.pos_ in {"ADJ"}:
            candidates.append((token.idx, token.idx + len(token.text), token.text))
    candidates = _unique_tokens(candidates)
    if not candidates:
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            if token.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}:
                candidates.append((token.idx, token.idx + len(token.text), token.text))
    candidates = _unique_tokens(candidates)
    if not candidates:
        return None
    start, end, text = rng.choice(candidates) if rng else candidates[0]
    return text, start, end


def build_missing_word_sample(
    caption: str,
    mask_token: str = "[MASK]",
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, str]]:
    selection = select_key_span_spacy(caption, rng=rng)
    if not selection:
        return None
    word, start, end = selection
    masked_caption = f"{caption[:start]}{mask_token}{caption[end:]}"
    question =  (f"<image>\nCaption: {masked_caption} \n"
                    "In the caption above, there is ONE missing word marked as "
                    f"{mask_token}. Please find out the missing word. "
    )
    return {
        "caption": caption,
        "masked_caption": masked_caption,
        "missing_word": word,
        "question": question,
    }


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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
) -> None:
    rng = random.Random(seed)
    records: List[Dict] = []
    for index, row in enumerate(_iter_jsonl(input_path)):
        caption = row["conversations"][1]["value"]
        image_rel = row.get("image")
        if not image_rel:
            continue
        image_path = os.path.join(image_root, image_rel)
        if not os.path.exists(image_path):
            continue
        sample = build_missing_word_sample(caption, mask_token=mask_token, rng=rng)
        if not sample:
            continue
        masked_caption = sample["masked_caption"]
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
                    "\n\n",
                    r"Output format: <think>[thinking process]</think>\boxed{[answer]}"
                ),
                "role": "system",
            },
            {
                "content": sample["question"],
                "role": "user",
            },
        ]
        payload = {
            "data_source": data_source,
            "prompt": prompt,
            "images": [image_path],
            "ability": "caption",
            "reward_model": {"ground_truth": sample["missing_word"], "style": "rule"},
            "extra_info": {
                "answer": sample["missing_word"],
                "index": index,
                "question": sample["question"],
                "split": split,
            },
        }
        records.append(payload)
        if len(records) >= max_samples:
            break
    if not records:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(records).to_parquet(output_path, index=False)
        os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
        pd.DataFrame(records).to_parquet(val_output_path, index=False)
        return
    val_size = max(1, int(round(len(records) * 0.02)))
    split_idx = max(0, len(records) - val_size)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(train_records).to_parquet(output_path, index=False)
    os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
    pd.DataFrame(val_records).to_parquet(val_output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create missing-word prompts from COCO captions."
    )
    parser.add_argument("--input", required=True, help="Input jsonl path.")
    parser.add_argument(
        "--output",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/train.parquet",
        help="Train parquet path.",
    )
    parser.add_argument(
        "--val-output",
        default="/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/validation.parquet",
        help="Validation parquet path.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples.")
    parser.add_argument("--mask-token", default="[MASK]", help="Mask token.")
    parser.add_argument(
        "--data-source", default="coco_karpathy_50k", help="Data source label."
    )
    parser.add_argument("--split", default="train", help="Split label.")
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
    )


if __name__ == "__main__":
    main()



'''
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_rlp_v0.py \
    --input /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco/annotations/coco_karpathy_train_50k.jsonl \
    --output /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/train.parquet \
    --max_samples 5000 


# sample format
  data_source: coco_karpathy_50k
  prompt: [{'content': "You are an AI assistant that rigorously follows this response protocol:\n\n1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.\n\n2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.\n\nEnsure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.", 'role': 'system'}
 {'content': '<image>\nCaption: {masked_caption} There is ONE missing word in the caption marked as [MASK]. Please find out the missing word and put your answer within <answer> and </answer> tags.', 'role': 'user'}]
  images: [{'bytes': None, 'path': 'MMPR-Tiny/images/8034_0.png'}]
  ability: caption
  reward_model: {'ground_truth': 'blue', 'style': 'rule'}
  extra_info: {'answer': 'blue', 'index': 8034, 'question': '<image>\nThere is ONE missing word in the caption marked as [MASK]. Please find out the missing word and put your answer within <answer> and </answer> tags.', 'split': 'train'}

'''