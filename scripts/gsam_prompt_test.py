"""POST the scene / grasp image to the running GSAM server with several prompts and print what it selects.

usage: python scripts/gsam_prompt_test.py datasets/<task> ["prompt a" "prompt b" ...]  (GSAM server must be running)"""
import base64, sys, requests
from pathlib import Path
T = Path(sys.argv[1])
URL = "http://127.0.0.1:6001/predict"
def b64(p): return base64.b64encode(p.read_bytes()).decode()
prompts = sys.argv[2:] or ["toilet paper", "toilet paper roll", "roll of toilet paper", "paper roll", "tissue roll", "white paper roll"]
for img, include_hand in [("scene_image.png", False), ("generated_human_grasp.png", True)]:
    print(f"\n### {img} (include_hand={include_hand})")
    data = b64(T / img)
    for p in prompts:
        r = requests.post(URL, json={"image_b64": data, "text_prompt": p, "include_hand": include_hand}, timeout=120).json()
        objs = [d for d in r.get("detections", []) if not d["is_hand"]]
        hands = [d for d in r.get("detections", []) if d["is_hand"]]
        o = objs[0] if objs else None
        box = [round(v) for v in o["bbox"]] if o else None
        w = (box[2]-box[0]) if box else None
        print(f"{p!r:24} -> obj={o['class_name'] if o else None!r:22} score={o['score'] if o else 0:.3f} bbox={box} width={w}"
              + (f"  hand={hands[0]['score']:.2f}" if hands else ("  hand=NONE" if include_hand else "")))
