import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
from PIL import Image


DEFAULT_DATA_PATH = (
    "/home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/datasets/rlp/v2/all.jsonl"
)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_data(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".parquet"):
        return pd.read_parquet(path).to_dict(orient="records")
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    raise ValueError(f"Unsupported file type: {path}")


def _get_image_path(sample: Dict[str, Any]) -> str:
    images = sample.get("images") or []
    if images:
        return images[0]
    extra = sample.get("extra_info") or {}
    paths = extra.get("image_paths") or []
    return paths[0] if paths else ""


def _read_image(path: str):
    if not path or not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")


def _extract_prompt(sample: Dict[str, Any]) -> Tuple[str, str]:
    prompt = sample.get("prompt") or []
    system_msg = next((p.get("content", "") for p in prompt if p.get("role") == "system"), "")
    user_msg = next((p.get("content", "") for p in prompt if p.get("role") == "user"), "")
    return system_msg, user_msg


def _get_sample(data: List[Dict[str, Any]], idx: int):
    if not data:
        return (None,) * 13
    idx = max(0, min(idx, len(data) - 1))
    sample = data[idx]
    extra = sample.get("extra_info") or {}
    reward = sample.get("reward_model") or {}
    system_msg, user_msg = _extract_prompt(sample)
    image_path = _get_image_path(sample)
    return (
        _read_image(image_path),
        str(idx),
        sample.get("data_source", ""),
        extra.get("split", ""),
        image_path,
        extra.get("masked_caption", ""),
        extra.get("answer", ""),
        extra.get("pos_tag", ""),
        ", ".join(extra.get("candidate_words") or []),
        extra.get("question", ""),
        json.dumps(reward.get("ground_truth", ""), ensure_ascii=False),
        system_msg,
        user_msg,
    )


def _load_action(path: str):
    data = _load_data(path)
    total = len(data)
    sample = _get_sample(data, 0)
    return (
        data,
        total,
        gr.update(maximum=total - 1 if total else 0, value=0),
        *sample,
    )


def _step_index(data: List[Dict[str, Any]], idx: int, delta: int):
    if not data:
        return (idx, *_get_sample(data, idx))
    idx = max(0, min(idx + delta, len(data) - 1))
    return (idx, *_get_sample(data, idx))


def build_ui(default_path: str):
    with gr.Blocks() as demo:
        gr.Markdown("## RLP v2 Dataset Viewer")
        with gr.Row():
            path = gr.Textbox(label="Dataset path", value=default_path)
            load_btn = gr.Button("Load")
            total = gr.Number(label="Total records", value=0, precision=0)
        data_state = gr.State([])
        idx = gr.Slider(0, 0, step=1, label="Index", value=0)
        with gr.Row():
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")
        with gr.Row():
            image = gr.Image(label="Image", type="pil")
            with gr.Column():
                index_text = gr.Textbox(label="Index", interactive=False)
                data_source = gr.Textbox(label="Data source", interactive=False)
                split = gr.Textbox(label="Split", interactive=False)
                image_path = gr.Textbox(label="Image path", interactive=False)
        masked_caption = gr.Textbox(label="Masked caption", interactive=False)
        answer = gr.Textbox(label="Selected word", interactive=False)
        pos_tag = gr.Textbox(label="POS tag", interactive=False)
        candidate_words = gr.Textbox(label="Candidate words", interactive=False)
        question = gr.Textbox(label="Question", interactive=False)
        reward_gt = gr.Textbox(label="Reward ground_truth", interactive=False)
        system_prompt = gr.Textbox(label="System prompt", interactive=False)
        user_prompt = gr.Textbox(label="User prompt", interactive=False)

        load_btn.click(
            _load_action,
            inputs=[path],
            outputs=[
                data_state,
                total,
                idx,
                image,
                index_text,
                data_source,
                split,
                image_path,
                masked_caption,
                answer,
                pos_tag,
                candidate_words,
                question,
                reward_gt,
                system_prompt,
                user_prompt,
            ],
        )
        idx.change(
            lambda data, i: _get_sample(data, int(i)),
            inputs=[data_state, idx],
            outputs=[
                image,
                index_text,
                data_source,
                split,
                image_path,
                masked_caption,
                answer,
                pos_tag,
                candidate_words,
                question,
                reward_gt,
                system_prompt,
                user_prompt,
            ],
        )
        prev_btn.click(
            lambda data, i: _step_index(data, int(i), -1),
            inputs=[data_state, idx],
            outputs=[idx, image, index_text, data_source, split, image_path, masked_caption,
                     answer, pos_tag, candidate_words, question, reward_gt, system_prompt, user_prompt],
        )
        next_btn.click(
            lambda data, i: _step_index(data, int(i), 1),
            inputs=[data_state, idx],
            outputs=[idx, image, index_text, data_source, split, image_path, masked_caption,
                     answer, pos_tag, candidate_words, question, reward_gt, system_prompt, user_prompt],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RLP v2 dataset with Gradio.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to jsonl/parquet.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo = build_ui(args.data)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()

