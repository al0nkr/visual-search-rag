# create_calib.py
from datasets import load_dataset
import json, os, requests
from PIL import Image
from io import BytesIO

ds = load_dataset("RIW/small-coco", split="validation[:512]")
out = []
for sample in ds:
    url  = sample["url"]        # COCO url
    cap  = sample["caption"]
    img  = sample["image"]
    fn   = f"calib_imgs/{sample['key']}.jpg"
    os.makedirs("calib_imgs", exist_ok=True)
    img.save(fn)
    out.append({"image": fn, "text": cap})

with open("calib_512.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)