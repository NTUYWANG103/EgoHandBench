"""EgoHandICL retrieval cues: VLM hand-involvement type, VLM caption and Qwen-7B text states.

Prompts are copied from the paper's Appendix B.1 (pre-defined visual templates)
and B.2 (reasoning-style adaptive textual templates). The paper uses
Qwen2.5-VL-72B-Instruct; this integration uses Qwen2.5-VL-7B-Instruct with greedy
decoding to keep per-frame inference tractable. The text encoder is Qwen-7B, as in the
paper and the released model file.
"""

import re

import numpy as np
import torch

TYPE_SYSTEM_PROMPT = """
You are an image understanding agent. Your task is to analyze a first-person perspective image and classify the interaction status of the left and right hands.

Classification rules:
- 0: Only the left hand is interacting with an object
- 1: Only the right hand is interacting with an object
- 2: Both hands are interacting with an object
- 3: Neither hand is interacting with any object

Important notes:
- Occlusion caused by objects must be considered in determining whether a hand is interacting.
- Use your best reasoning based on the visual content to make this decision.

Your response must be a single number: one of [0, 1, 2, -1]. Do not include any explanation or additional text.
"""

TYPE_USER_PROMPT = """
Please analyze the first-person perspective image and determine the interaction status of the hands.
Return only the correct label number based on the following:
- 0: Only the left hand is interacting
- 1: Only the right hand is interacting
- 2: Both hands are interacting
- 3: Neither hand is interacting
"""

CAPTION_SYSTEM_PROMPT = """
You are a reasoning agent specialized in egocentric hand-object interaction understanding.
Your task is to analyze images captured from a first-person perspective and generate a caption that describes all hand and object interaction details relevant to 3D hand reconstruction.
Given a single input image, output one concise sentence that specifies which hand(s) (left/right/both) are visible, their pose, and their interaction with any objects.
Be precise and descriptive about the hand-object interaction without adding explanations or speculations.
"""

CAPTION_USER_PROMPT = """
Please analyze the image and return one sentence describing the hands and their interactions with objects for hand reconstruction (e.g., left hand grasping a cup, right hand resting on the table).
"""

# Image tokens per frame: 256 merged 28x28 patches.
VLM_MAX_PIXELS = 256 * 28 * 28
VLM_MIN_PIXELS = 64 * 28 * 28
CAPTION_TOKENS = 48
# The paper's prompt lists class 3 for "neither hand" but asks for an answer in [0, 1, 2, -1]; the released
# labels use 3 for that class (-1 appears 17 times in 168,931), so -1 means 3. Unparseable answers fall back
# to the most common class (both hands).
DEFAULT_TYPE = 2


def parse_type(text):
    match = re.search(r"-?\d", text)
    value = int(match.group(0)) if match else DEFAULT_TYPE
    value = 3 if value == -1 else value
    return value if value in (0, 1, 2, 3) else DEFAULT_TYPE


class RetrievalCues:
    """Qwen2.5-VL classification and caption for full RGB frames."""

    def __init__(self, path, device):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            path, min_pixels=VLM_MIN_PIXELS, max_pixels=VLM_MAX_PIXELS
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(device).eval()

    def _generate(self, images, system, user, max_new_tokens):
        from PIL import Image

        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
            ]
        ] * len(images)
        prompts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = self.processor(
            text=prompts, images=[Image.fromarray(x) for x in images],
            padding=True, return_tensors="pt",
        ).to(self.device)
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
        output = output[:, inputs["input_ids"].shape[1]:]
        return [
            s.strip() for s in self.processor.batch_decode(output, skip_special_tokens=True)
        ]

    @torch.inference_mode()
    def __call__(self, images, batch_size=16):
        """Return hand-involvement types and captions for RGB uint8 frames."""
        types, captions = [], []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            types += [
                parse_type(t)
                for t in self._generate(chunk, TYPE_SYSTEM_PROMPT, TYPE_USER_PROMPT, 4)
            ]
            captions += self._generate(
                chunk, CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT, CAPTION_TOKENS
            )
        return np.asarray(types, np.int64), captions


class TextEncoder:
    """Qwen-7B last hidden states, one unpadded call per text as in released inference."""

    def __init__(self, path, device):
        # Qwen-7B's own remote code, bundled locally (qwen/), instead of trust_remote_code.
        from .qwen.configuration_qwen import QWenConfig
        from .qwen.modeling_qwen import QWenLMHeadModel
        from .qwen.tokenization_qwen import QWenTokenizer

        self.tokenizer = QWenTokenizer.from_pretrained(path)
        self.tokenizer.eos_token = "<|endoftext|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        config = QWenConfig.from_pretrained(path, bf16=True)
        self.model = QWenLMHeadModel.from_pretrained(
            path, config=config, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0]
