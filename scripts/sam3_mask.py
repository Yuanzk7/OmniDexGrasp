"""scene_image.png + text prompt -> binary mask (same flow as sam3/test_image.py)."""
import sys
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

image_path, prompt, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
model = build_sam3_image_model()
processor = Sam3Processor(model)
image = Image.open(image_path).convert("RGB")
with torch.autocast("cuda", dtype=torch.bfloat16):
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)
masks, scores = output["masks"], output["scores"]
print("detections:", len(scores), "scores:", [round(float(s), 3) for s in scores])
best = masks[scores.argmax().item()].squeeze().cpu().numpy()
Image.fromarray(best.astype(np.uint8) * 255).save(out_path)
print("mask saved:", out_path, "fg ratio:", round(float(best.mean()), 3))
