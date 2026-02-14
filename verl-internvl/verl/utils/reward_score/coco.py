# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import re
import time
import threading

import requests
from mathruler.grader import extract_boxed_content, grade_answer


CLIP_REWARD_SCALE = 0.3
CLIP_RM_MAX_RETRIES = int(os.getenv("CLIP_RM_MAX_RETRIES", "3"))
CLIP_RM_TIMEOUT = float(os.getenv("CLIP_RM_TIMEOUT", "10"))
_CLIP_RM_ENDPOINTS_CACHE = None
_CLIP_RM_ENDPOINTS_LOCK = threading.Lock()
_CLIP_RM_ENDPOINT_IDX = 0


def _get_clip_rm_endpoints():
    endpoints_env = os.getenv("CLIP_RM_ENDPOINTS", "")
    if endpoints_env:
        endpoints = [e.strip().rstrip("/") for e in endpoints_env.split(",") if e.strip()]
        return [
            e if e.endswith("/get_reward") else f"{e}/get_reward"
            for e in endpoints
        ]

    hosts_env = os.getenv("CLIP_RM_HOSTS", "127.0.0.1")
    base_port = int(os.getenv("CLIP_RM_BASE_PORT", "5000"))
    num_rms = int(os.getenv("CLIP_RM_NUM_RMS", "1"))
    hosts = [h.strip() for h in hosts_env.split(",") if h.strip()]
    return [
        f"http://{host}:{base_port + idx}/get_reward"
        for host in hosts
        for idx in range(num_rms)
    ]


def _next_clip_rm_endpoint():
    global _CLIP_RM_ENDPOINTS_CACHE, _CLIP_RM_ENDPOINT_IDX
    if _CLIP_RM_ENDPOINTS_CACHE is None:
        _CLIP_RM_ENDPOINTS_CACHE = _get_clip_rm_endpoints()
    if not _CLIP_RM_ENDPOINTS_CACHE:
        raise RuntimeError("No CLIP RM endpoints configured.")
    return random.choice(_CLIP_RM_ENDPOINTS_CACHE)
    with _CLIP_RM_ENDPOINTS_LOCK:
        endpoint = _CLIP_RM_ENDPOINTS_CACHE[_CLIP_RM_ENDPOINT_IDX % len(_CLIP_RM_ENDPOINTS_CACHE)]
        _CLIP_RM_ENDPOINT_IDX += 1
    return endpoint


def get_clip_reward(image_path, text):
    payload = {"image_path": image_path, "text": text}
    last_error = None
    for attempt in range(CLIP_RM_MAX_RETRIES):
        endpoint = _next_clip_rm_endpoint()
        try:
            response = requests.post(endpoint, json=payload, timeout=CLIP_RM_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if "reward" not in data:
                raise ValueError(f"Invalid response: {data}")
            return float(data["reward"])
        except Exception as exc:
            last_error = exc
            time.sleep(0.2 * (2 ** attempt))
    raise RuntimeError(f"CLIP RM API failed after {CLIP_RM_MAX_RETRIES} attempts: {last_error}")



def get_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def get_acc_reward(
    predict_str: str, ground_truth: str, use_boxed: bool = True,
    rule_based_only: bool = False,
    **kwargs
    ) -> float:
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    model_used = False
    info = {
        'gt_img_similarity': -1,
        'pred_img_similarity': -1,
        'ratio': 0,
        'raw_reward': 0.0,
        'model_used': float(model_used),
    }
    if use_boxed:
        answer = extract_boxed_content(predict_str)
        if answer == 'None':
            return info
    else:
        answer = predict_str

    for gt in ground_truth:
        graded = grade_answer(answer, gt)
        if graded:
            info = {
                'gt_img_similarity': 1.,
                'pred_img_similarity': 1.,
                'ratio': 1.,
                'raw_reward': 1.,
                'model_used': float(model_used),
            }
            return info


    if rule_based_only:
        return info
    # CLIP reward
    image = kwargs.get("image", None)

    missing_caption = kwargs.get("missing_caption", '')
    gt_caption = missing_caption.replace('[MASK]', ground_truth[0])
    pred_caption = missing_caption.replace("[MASK]", answer)

    pred_caption_clip_reward = get_clip_reward(image, pred_caption)
    gt_caption_clip_reward = get_clip_reward(image, gt_caption)
    model_used = True

    ratio = pred_caption_clip_reward / gt_caption_clip_reward
    reward = min(ratio, 1.0)

    info = {
        'gt_img_similarity': gt_caption_clip_reward,
        'pred_img_similarity': pred_caption_clip_reward,
        'ratio': ratio,
        'raw_reward': reward,
        'model_used': float(model_used),
    }

    return info

    # return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    predict_str: str, 
    ground_truth: str, 
    use_boxed: bool = True, 
    format_score: float = 0.1,
    **kwargs
) -> float:

    extra_info = kwargs.get('extra_info', {})
    question = extra_info.get('question', '')
    pos_tag = extra_info.get('pos_tag', '')
    missing_caption = question.split('Caption: ')[1].strip().split('In the caption above, ')[0].strip()

    assert '[MASK]' in missing_caption, f"Missing caption is not found in the question: {question}"

    input_info = {
        'image': extra_info['image_paths'][0],
        'missing_caption': missing_caption,
    }
    
    rule_based_only = False
    if pos_tag == 'number':
        rule_based_only = True

    computed_reward = get_acc_reward(
        predict_str, ground_truth, use_boxed, 
        rule_based_only=rule_based_only,
        **input_info
    )
    if computed_reward['model_used']:
        # computed_reward is a dict
        acc_reward = computed_reward['raw_reward'] * CLIP_REWARD_SCALE
    else:
        # computed_reward is a float
        acc_reward = computed_reward['raw_reward']

    score = (1.0 - format_score) * acc_reward + format_score * get_format_reward(predict_str)

    ret = {
        'score': score,
        **computed_reward,
    }
    return ret
