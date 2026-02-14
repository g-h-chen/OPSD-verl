import argparse
import threading

from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import SiglipModel, SiglipProcessor

app = Flask(__name__)

# Argument parsing (keep legacy args for compatibility with launch scripts)
parser = argparse.ArgumentParser(description="SigLIP reward service.")
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
parser.add_argument("--siglip_model_name", type=str, default="google/siglip-base-patch16-224")
parser.add_argument("--test_image_path", type=str, default=None, help="Run a one-off test on this image.")
parser.add_argument("--test_text", type=str, default=None, help="Run a one-off test with this text.")
# Legacy args (ignored but accepted)
parser.add_argument("--reward_mode", type=str, default=None)
parser.add_argument("--aha_path", type=str, default=None)
parser.add_argument("--model_dir", type=str, default=None)
args = parser.parse_args()

# Load SigLIP model for image-text reward
siglip_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
siglip_model = SiglipModel.from_pretrained(args.siglip_model_name).to(siglip_device)
siglip_processor = SiglipProcessor.from_pretrained(args.siglip_model_name)
siglip_lock = threading.Lock()
concurrency_limiter = threading.Semaphore(args.max_concurrent_queries)
print(f"SigLIP model loaded from {args.siglip_model_name} on {siglip_device}!!!")


def _compute_similarity(image_path: str, text: str) -> float:
    image = Image.open(image_path).convert("RGB")
    inputs = siglip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(siglip_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = siglip_model(**inputs)
    prob = torch.sigmoid(outputs.logits_per_image[0, 0])
    return float(prob.item())


@app.route("/siglip_reward", methods=["POST"])
def siglip_reward():
    data = request.json or {}
    if data.get("test_connection", None):
        return jsonify({"message": "Connection is fine"}), 200
    image_path = data.get("image_path")
    text = data.get("text")
    if not image_path or text is None:
        return jsonify({"error": "Missing image_path or text"}), 400
    text_preview = (text[:80] + "...") if len(text) > 80 else text
    print(f"[siglip_reward] from={request.remote_addr} image={image_path} text='{text_preview}'")
    concurrency_limiter.acquire()
    try:
        with siglip_lock:
            reward = _compute_similarity(image_path, text)
        print(f"[siglip_reward] reward={reward:.6f}")
        return jsonify({"reward": reward})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        concurrency_limiter.release()


if __name__ == "__main__":
    if args.test_image_path and args.test_text:
        with siglip_lock:
            sim = _compute_similarity(args.test_image_path, args.test_text)
        print(f"siglip_sim={sim:.6f}")
        raise SystemExit(0)
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)

'''

python /home/efs/hardychen/workspaces/gptoss_rlp/verl-internvl/verl/utils/reward_score/rms/siglip_rm.py --test_image_path /home/efs/hardychen/workspaces/gptoss_rlp/InternVL/internvl_chat_gpt_oss/data/coco/train2014/COCO_train2014_000000283122.jpg --test_text "Two motorcycles parked on a shiny showroom floor inside a retro garage filled with vintage gas pumps, neon beer signs, and old automotive memorabilia."

'''




