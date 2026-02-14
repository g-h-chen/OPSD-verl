import argparse
import json
import os
import random
import string
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

_PPL_TOKENIZER = None
_PPL_MODEL = None
_PPL_DEVICE = None
_PPL_BASE_PROMPT = None
_WORDNET = None
_POS_TAGGER = None

_INTERNVL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "InternVL", "internvl_chat")
)
if _INTERNVL_ROOT not in sys.path:
    sys.path.append(_INTERNVL_ROOT)

from internvl.conversation import get_conv_template  # type: ignore
from internvl.train.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios, width: int, height: int, image_size: int
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num: int = 1, max_num: int = 8, image_size: int = 448, use_thumbnail: bool = False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


def _load_internvl(
    model_id: str = "OpenGVLab/InternVL3_5-8B-Pretrained",
    device: Optional[str] = None,
):
    global _PPL_TOKENIZER, _PPL_MODEL, _PPL_DEVICE
    if _PPL_TOKENIZER is None or _PPL_MODEL is None:
        _PPL_TOKENIZER = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _PPL_DEVICE = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _PPL_MODEL = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        _PPL_MODEL.eval()
    return None, _PPL_TOKENIZER, _PPL_MODEL, _PPL_DEVICE


def _build_prompt(tokenizer, model, caption: str, num_patches: int) -> Tuple[str, int, int]:
    global _PPL_BASE_PROMPT
    question = "<image>\nProvide a one-sentence caption for the provided image."
    if _PPL_BASE_PROMPT is None:
        template = get_conv_template(model.template)
        template.system_message = getattr(model, "system_message", "")
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        _PPL_BASE_PROMPT = template.get_prompt()
    template_with_caption = get_conv_template(model.template)
    template_with_caption.system_message = getattr(model, "system_message", "")
    template_with_caption.append_message(template_with_caption.roles[0], question)
    template_with_caption.append_message(template_with_caption.roles[1], caption)
    full_prompt = template_with_caption.get_prompt()
    image_tokens = (
        IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    )
    prompt = _PPL_BASE_PROMPT.replace("<image>", image_tokens, 1)
    full_prompt = full_prompt.replace("<image>", image_tokens, 1)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    caption_len = tokenizer(caption, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
    return full_prompt, prompt_len, caption_len


def _is_punct_word(word: str) -> bool:
    return bool(word) and all(ch in string.punctuation for ch in word)


def _load_wordnet():
    global _WORDNET
    if _WORDNET is None:
        try:
            import nltk  # type: ignore[import-not-found]
            from nltk.corpus import wordnet as wn  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "nltk is required for abstract-word filtering. "
                "Install it with: pip install nltk"
            ) from exc
        try:
            wn.synsets("dog")
        except LookupError as exc:
            raise RuntimeError(
                "nltk wordnet data is missing. Run: "
                "python -m nltk.downloader wordnet omw-1.4"
            ) from exc
        _WORDNET = wn
    return _WORDNET


def _load_pos_tagger():
    global _POS_TAGGER
    if _POS_TAGGER is None:
        try:
            import nltk  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "nltk is required for POS tagging. Install it with: pip install nltk"
            ) from exc
        try:
            nltk.pos_tag(["test"])
        except LookupError as exc:
            raise RuntimeError(
                "nltk tagger data is missing. Run: "
                "python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng"
            ) from exc
        _POS_TAGGER = nltk.pos_tag
    return _POS_TAGGER


def _get_pos_tags(words: List[str]) -> List[str]:
    tagger = _load_pos_tagger()
    return [tag for _, tag in tagger(words)]


_CONCRETE_ROOTS = {
    "physical_entity.n.01",
    "object.n.01",
    "artifact.n.01",
    "person.n.01",
    "animal.n.01",
}


def _is_concrete_noun_synset(synset) -> bool:
    for path in synset.hypernym_paths():
        for node in path:
            if node.name() in _CONCRETE_ROOTS:
                return True
    return False


def _is_concrete_word(word: str) -> bool:
    if _is_punct_word(word):
        return False
    normalized = word.strip(string.punctuation).lower()
    if not normalized:
        return False
    wn = _load_wordnet()
    lemma = wn.morphy(normalized, wn.VERB) or normalized
    if lemma == "be":
        return False
    if wn.synsets(normalized, pos=wn.VERB):
        return False
    synsets = wn.synsets(normalized)
    if not synsets:
        return False
    for syn in synsets:
        if syn.pos() == "n" and _is_concrete_noun_synset(syn):
            return True
    for syn in synsets:
        if syn.pos() in {"a", "s", "r"}:
            for lemma in syn.lemmas():
                for related in lemma.derivationally_related_forms():
                    for noun_syn in wn.synsets(related.name(), pos=wn.NOUN):
                        if _is_concrete_noun_synset(noun_syn):
                            return True
    return False

def _word_spans(caption: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    words = [word for word in caption.split(" ") if word]
    spans: List[Tuple[int, int]] = []
    idx = 0
    for word in words:
        while idx < len(caption) and caption[idx] == " ":
            idx += 1
        start = idx
        end = start + len(word)
        spans.append((start, end))
        idx = end
    return words, spans


def _token_offsets(tokenizer, caption: str) -> List[Tuple[int, int]]:
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(caption, add_special_tokens=False, return_offsets_mapping=True)
        return [(start, end) for start, end in enc["offset_mapping"]]
    ids = tokenizer(caption, add_special_tokens=False).input_ids
    tokens = tokenizer.convert_ids_to_tokens(ids)
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for token in tokens:
        cleaned = token.lstrip("▁")
        if not cleaned:
            offsets.append((pos, pos))
            continue
        idx = caption.find(cleaned, pos)
        if idx < 0:
            idx = pos
        offsets.append((idx, idx + len(cleaned)))
        pos = idx + len(cleaned)
    return offsets


def _aggregate_word_ppl(
    token_ppl: List[float],
    token_offsets: List[Tuple[int, int]],
    word_spans: List[Tuple[int, int]],
) -> List[float]:
    per_word: List[List[float]] = [[] for _ in word_spans]
    for ppl, (t_start, t_end) in zip(token_ppl, token_offsets):
        for idx, (w_start, w_end) in enumerate(word_spans):
            if t_start < w_end and t_end > w_start:
                per_word[idx].append(ppl)
                break
    word_ppl: List[float] = []
    for values in per_word:
        word_ppl.append(sum(values) / len(values) if values else 0.0)
    return word_ppl


def get_word_ppl_batch(samples: List[Dict]) -> List[List[float]]:
    if not samples:
        return []
    image_processor, tokenizer, model, device = _load_internvl()
    pixel_values_list = []
    full_texts: List[str] = []
    prompt_lens: List[int] = []
    caption_lens: List[int] = []
    captions: List[str] = []
    for sample in samples:
        caption = sample.get("caption", "").strip()
        image_path = sample.get("image_path")
        if not caption or not image_path:
            pixel_values_list.append(None)
            full_texts.append("")
            prompt_lens.append(0)
            caption_lens.append(0)
            captions.append("")
            continue
        pixel_values = load_image(image_path, max_num=8)
        num_patches = pixel_values.shape[0]
        full_text, prompt_len, caption_len = _build_prompt(
            tokenizer, model, caption, num_patches=num_patches
        )
        pixel_values_list.append(pixel_values)
        full_texts.append(full_text)
        prompt_lens.append(prompt_len)
        caption_lens.append(caption_len)
        captions.append(caption)
    valid_indices = [i for i, pv in enumerate(pixel_values_list) if pv is not None]
    if not valid_indices:
        return [[] for _ in samples]
    pixel_values = torch.cat([pixel_values_list[i] for i in valid_indices], dim=0)
    model_inputs = tokenizer([full_texts[i] for i in valid_indices], return_tensors="pt", padding=True)
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    pixel_values = pixel_values.to(device, dtype=model.dtype)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    image_flags = torch.ones((pixel_values.shape[0],), dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
        )
        logits = outputs.logits
    target_ids = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    results: List[List[float]] = [[] for _ in samples]
    for batch_idx, sample_idx in enumerate(valid_indices):
        caption = captions[sample_idx]
        if not caption:
            continue
        start_idx = max(prompt_lens[sample_idx] - 1, 0)
        end_idx = min(start_idx + caption_lens[sample_idx], token_log_probs.shape[1])
        caption_log_probs = token_log_probs[batch_idx, start_idx:end_idx]
        token_ppl = torch.exp(-caption_log_probs).tolist()
        token_offsets = _token_offsets(tokenizer, caption)
        if len(token_offsets) != len(token_ppl):
            min_len = min(len(token_offsets), len(token_ppl))
            token_offsets = token_offsets[:min_len]
            token_ppl = token_ppl[:min_len]
        words, word_spans = _word_spans(caption)
        word_ppl = _aggregate_word_ppl(token_ppl, token_offsets, word_spans)
        for idx, word in enumerate(words):
            if _is_punct_word(word):
                word_ppl[idx] = 0.0
        results[sample_idx] = word_ppl
    return results


def get_word_ppl(sample_dict: Dict) -> List[float]:
    result = get_word_ppl_batch([sample_dict])
    return result[0] if result else []


def _candidate_indices(words: List[str], pos_tags: List[str]) -> List[int]:
    candidates: List[int] = []
    for idx in range(1, len(words)):
        if _is_punct_word(words[idx]):
            continue
        if pos_tags[idx] in {"IN", "TO"} or pos_tags[idx].startswith("VB"):
            continue
        if not _is_concrete_word(words[idx]):
            continue
        candidates.append(idx)
    return candidates


def _select_from_word_ppl(
    words: List[str],
    spans: List[Tuple[int, int]],
    word_ppl: List[float],
    candidate_indices: List[int],
) -> Optional[Tuple[str, int, int, float]]:
    if not word_ppl or len(word_ppl) != len(words):
        return None
    filtered = [idx for idx in candidate_indices if 15 <= word_ppl[idx] <= 40]
    if not filtered:
        return None
    max_idx = max(filtered, key=word_ppl.__getitem__)
    start, end = spans[max_idx]
    return words[max_idx], start, end, word_ppl[max_idx]


def select_key_span_ppl(
    caption: str,
    image_path: str,
) -> Optional[Tuple[str, int, int, float]]:
    if not caption or not image_path:
        return None
    words, spans = _word_spans(caption)
    if len(words) < 2:
        return None
    pos_tags = _get_pos_tags(words)
    candidate_indices = _candidate_indices(words, pos_tags)
    if not candidate_indices:
        return None
    word_ppl = get_word_ppl({"caption": caption, "image_path": image_path})
    return _select_from_word_ppl(words, spans, word_ppl, candidate_indices)


def build_missing_word_sample(
    caption: str,
    image_path: str,
    mask_token: str = "[MASK]",
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, str]]:
    _ = rng
    selection = select_key_span_ppl(caption, image_path)
    if not selection:
        return None
    word, start, end, masked_ppl = selection
    masked_caption = f"{caption[:start]}{mask_token}{caption[end:]}"
    question =  (f"<image>\nCaption: {masked_caption} \n"
                    "In the caption above, there is ONE missing word marked as "
                    f"{mask_token}. Please find out the missing word. "
    )
    return {
        "caption": caption,
        "masked_caption": masked_caption,
        "missing_word": word,
        "masked_ppl": masked_ppl,
        "question": question,
    }


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _process_rows(
    rows: List[Dict],
    shard_id: int,
    mask_token: str,
    data_source: str,
    split: str,
    image_root: str,
    device: Optional[str] = None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    shared_count: Optional["torch.multiprocessing.Value"] = None,
    count_lock: Optional["torch.multiprocessing.Lock"] = None,
    stop_event: Optional["torch.multiprocessing.Event"] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    if device is not None:
        torch.cuda.set_device(device)
    records: List[Dict] = []
    scanned = 0
    kept = 0
    pending_samples: List[Dict] = []
    pending_meta: List[Dict] = []

    def _flush_pending() -> None:
        nonlocal kept
        if not pending_samples:
            return
        word_ppl_list = get_word_ppl_batch(pending_samples)
        for sample, meta, word_ppl in zip(pending_samples, pending_meta, word_ppl_list):
            selection = _select_from_word_ppl(
                meta["words"], meta["spans"], word_ppl, meta["candidate_indices"]
            )
            if not selection:
                continue
            word, start, end, masked_ppl = selection
            caption = meta["caption"]
            masked_caption = f"{caption[:start]}{mask_token}{caption[end:]}"
            question = (
                f"<image>\nCaption: {masked_caption} \n"
                "In the caption above, there is ONE missing word marked as "
                f"{mask_token}. Please find out the missing word. "
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
                        r"to the user's question within \boxed{}.\n\n"
                        r"Output format: <think>[thinking process]</think>\boxed{[answer]}"
                    ),
                    "role": "system",
                },
                {
                    "content": question,
                    "role": "user",
                },
            ]
            payload = {
                "data_source": data_source,
                "prompt": prompt,
                "images": [meta["image_path"]],
                "ability": "caption",
                "reward_model": {"ground_truth": word, "style": "rule"},
                "extra_info": {
                    "answer": word,
                    "index": meta["index"],
                    "masked_ppl": masked_ppl,
                    "question": question,
                    "split": split,
                    "image_paths": [meta["image_path"]],
                },
            }
            records.append(payload)
            kept += 1
            if kept % 50 == 0:
                print(f"[shard {shard_id}] collected {kept} samples")
            if max_samples:
                if shared_count is not None and count_lock is not None:
                    with count_lock:
                        shared_count.value += 1
                        if shared_count.value >= max_samples and stop_event is not None:
                            stop_event.set()
                            return
                elif len(records) >= max_samples:
                    return

    for index, row in enumerate(tqdm(rows, desc=f"Shard {shard_id}", position=shard_id)):
        if stop_event is not None and stop_event.is_set():
            break
        scanned += 1
        caption = row["conversations"][1]["value"]
        image_rel = row.get("image")
        if not image_rel:
            continue
        image_path = os.path.join(image_root, image_rel)
        if not os.path.exists(image_path):
            continue
        words, spans = _word_spans(caption)
        if len(words) < 2:
            continue
        pos_tags = _get_pos_tags(words)
        candidate_indices = _candidate_indices(words, pos_tags)
        if not candidate_indices:
            continue
        pending_samples.append({"caption": caption, "image_path": image_path})
        pending_meta.append(
            {
                "caption": caption,
                "image_path": image_path,
                "index": index,
                "words": words,
                "spans": spans,
                "candidate_indices": candidate_indices,
            }
        )
        if len(pending_samples) >= batch_size:
            _flush_pending()
            pending_samples.clear()
            pending_meta.clear()
            if max_samples:
                if shared_count is not None and count_lock is not None:
                    with count_lock:
                        if shared_count.value >= max_samples:
                            if stop_event is not None:
                                stop_event.set()
                            break
                elif len(records) >= max_samples:
                    break
    _flush_pending()
    return records, {"scanned": scanned, "kept": kept}


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
    batch_size: int = 4,
) -> None:
    rng = random.Random(seed)
    _ = rng
    if summary_json_path is None:
        # base, _ext = os.path.splitext(output_path)
        # summary_json_path = f"{base}.summary.json"
        summary_json_path = os.path.join(os.path.dirname(output_path), "summary.json")
    rows = list(_iter_jsonl(input_path))
    num_gpus = min(8, torch.cuda.device_count())
    stats = {"scanned": 0, "kept": 0}
    # num_gpus = 1
    if num_gpus <= 1:
        records, stats = _process_rows(
            rows,
            shard_id=0,
            mask_token=mask_token,
            data_source=data_source,
            split=split,
            image_root=image_root,
            device="cuda:0" if torch.cuda.is_available() else None,
            batch_size=batch_size,
            max_samples=max_samples,
        )
    else:
        shard_size = (len(rows) + num_gpus - 1) // num_gpus
        shards = [
            rows[i * shard_size : (i + 1) * shard_size] for i in range(num_gpus)
        ]
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Manager() as manager:
            shared_count = manager.Value("i", 0)
            count_lock = manager.Lock()
            stop_event = manager.Event()
            with ctx.Pool(processes=num_gpus) as pool:
                results = pool.starmap(
                    _process_rows,
                    [
                        (
                            shards[i],
                            i,
                            mask_token,
                            data_source,
                            split,
                            image_root,
                            f"cuda:{i}",
                        batch_size,
                            max_samples,
                            shared_count,
                            count_lock,
                            stop_event,
                        )
                        for i in range(num_gpus)
                    ],
                )
        records = [record for shard, _ in results for record in shard]
        stats["scanned"] = sum(meta["scanned"] for _, meta in results)
        stats["kept"] = sum(meta["kept"] for _, meta in results)
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
                        "num_gpus": num_gpus,
                        "scanned_rows": stats["scanned"],
                        "kept_rows": stats["kept"],
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
                    "num_gpus": num_gpus,
                    "scanned_rows": stats["scanned"],
                    "kept_rows": stats["kept"],
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
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU worker.")
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
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()



'''
python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/scripts/data_preprocessing/collate_rlp_v1.py \
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