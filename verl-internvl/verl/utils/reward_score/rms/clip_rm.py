import argparse
import threading

from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

app = Flask(__name__)

# Argument parsing (keep legacy args for compatibility with launch scripts)
parser = argparse.ArgumentParser(description="CLIP reward service.")
parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on.")
parser.add_argument(
    "--max_concurrent_queries",
    type=int,
    default=1,
    help="Maximum concurrent requests.",
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help='Device to run the model on (e.g., "cuda" or "cpu").',
)
parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14-336")
# Legacy args (ignored but accepted)
parser.add_argument("--reward_mode", type=str, default=None)
parser.add_argument("--aha_path", type=str, default=None)
parser.add_argument("--model_dir", type=str, default=None)
args = parser.parse_args()

# Load CLIP model for image-text reward
clip_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(clip_device)
clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
clip_lock = threading.Lock()
concurrency_limiter = threading.Semaphore(args.max_concurrent_queries)
print(f"CLIP model loaded from {args.clip_model_name} on {clip_device}!!!")


def test_texts_one_image(
    image_path: str,
    *texts: str,
    max_length: int = 64,
) -> torch.Tensor:
    """
    Quick test: one image + N texts, return cosine similarities per text.
    """
    if not texts:
        raise ValueError("At least one text must be provided.")
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(
        text=list(texts),
        images=image,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(clip_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds.expand_as(outputs.text_embeds)
    similarities = torch.nn.functional.cosine_similarity(outputs.text_embeds, image_embeds)
    for idx, text in enumerate(texts):
        print(f"{similarities[idx]:.4f} similarity for image 0 and '{text}'")
    return similarities


@app.route("/get_reward", methods=["POST"])
def clip_reward():
    data = request.json or {}
    if data.get("test_connection", None):
        return jsonify({"message": "Connection is fine"}), 200
    image_path = data.get("image_path")
    text = data.get("text")
    if not image_path or text is None:
        return jsonify({"error": "Missing image_path or text"}), 400
    text_preview = (text[:80] + "...") if len(text) > 80 else text
    print(f"[clip_reward] from={request.remote_addr} image={image_path} text='{text_preview}'")
    concurrency_limiter.acquire()
    try:
        with clip_lock:
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(clip_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = clip_model(**inputs)
            similarity = torch.nn.functional.cosine_similarity(outputs.text_embeds, outputs.image_embeds)
            reward = float(similarity.item())
        print(f"[clip_reward] reward={reward:.6f}")
        return jsonify({"reward": reward})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        concurrency_limiter.release()


if __name__ == "__main__":
    # test_texts_one_image(
    #     '/home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco/train2014/COCO_train2014_000000243783.jpg',
    #     "A tennis ball sits on a [MASK] in the grass",
    #     "A tennis ball sits on a racket in the grass",

    # ); exit()

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)

