import torch, json, os
from datasets import Dataset
from PIL import Image
from transformers import HunYuanVLForConditionalGeneration, AutoProcessor
import base64
from io import BytesIO
import torch, json, os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import HunYuanVLForConditionalGeneration, AutoProcessor
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.core import CompressionSession, EventType

from llmcompressor import oneshot
import random

MODEL_ID   = "tencent/HunyuanOCR"
SAVE_DIR   = "./HunyuanOCR-AWQ-W4A16"
CALIB_FILE = "calib_512.json"

# 1.  load
model = HunYuanVLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            offload_buffers=True,
            attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# # 2.  raw HF dataset  (text=str, image=PIL)
# def build_ds():
#     data = json.load(open(CALIB_FILE))
#     return Dataset.from_list([
#         {"image": Image.open(d["image"]).convert("RGB"),
#          "text":  f"<image>{d['text']}"}      #  keep <image> for safety
#         for d in data
#     ])

# calib_ds = build_ds()

# # 3.  tell llm-compressor how to turn samples → tensors
# def collate(sample):
#     # 1.  build the prompt exactly like the model card
#     text = f"<image>\n{sample['text']}"          #  MUST contain <image>
#     # 2.  single processor call – do NOT tokenise beforehand
#     batch = processor(text=text,
#                       images=sample["image"],
#                       return_tensors="pt",
#                       truncation=True,
#                       max_length=2048)
#     return batch

# Oneshot arguments
DATASET_ID = "jxie/flickr8k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# ---------- 1.  build the SAME message format ----------
def preprocess(example):
    idx   = random.randint(0, 4)
    cap_key = f"caption_{idx}"
    caption = example.get(cap_key, example.get("caption", ""))
    if isinstance(caption, (list, tuple)):
        caption = " ".join(map(str, caption))

    #  HunyuanOCR expects exactly this structure
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},   #  PIL image object
                {"type": "text",  "text":  "What does this image show?"},
            ],
        },
        {"role": "assistant", "content": caption},
    ]
    #  processor returns token ids **and** pixel_values
    return {
        "text": processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
        "images": example["image"],
    }

ds = ds.map(preprocess)
print("Sample after preprocess:", ds[0])

def tokenize(sample):
    #  sample is dict of lists (batch_size=1) -> squeeze to 1-D
    texts = sample["text"]
    image_inputs = sample["images"]
    if isinstance(texts, (list, tuple)):
        texts = texts[0]
    if isinstance(image_inputs, (list, tuple)):
        image_inputs = image_inputs[0]

    enc = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # squeeze the batch dim (single-sample)
    return {
        k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.shape and v.shape[0] == 1 else v
        for k, v in enc.items()
    }

ds = ds.map(tokenize,   batched=False, remove_columns=ds.column_names)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    # batch: list[dict] -> use the processor's pad to handle variable-length sequences and stack tensors
    # This supports increasing the DataLoader batch_size > 1
    return processor.pad(batch, return_tensors="pt")


# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:model.vision_embed_tokens.*"],
)

session = CompressionSession()

# # Perform oneshot
# print("Dataset length:", len(ds))
# oneshot(
#     model=model,
#     dataset=ds,
#     recipe=recipe,
#     max_seq_length=MAX_SEQUENCE_LENGTH,
#     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#     trust_remote_code_model=True,
#     processor=processor,
#     data_collator=data_collator
# )

session.initialize(model=model, 
                   recipe=recipe, 
                   calib_data=ds, 
                   num_calibration_samples=512, 
                   max_seq_length=MAX_SEQUENCE_LENGTH, 
                   data_collator=data_collator, 
                   processor=processor, 
                   trust_remote_code_model=True)
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=lambda x: x)
session.event(EventType.CALIBRATION_EPOCH_START)
for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    session.event(EventType.BATCH_START)
    session.event(EventType.BATCH_END, batch_data=batch)   #  no loss needed
    if step >= 511:
        break
session.event(EventType.CALIBRATION_EPOCH_END)
session.finalize()
# # Confirm generations of the quantized model look sane.
# print("========== SAMPLE GENERATION ==============")
# # 1) Load or reuse any image (e.g. from your calib dataset or flickr8k)
# # Here I’ll assume you have a local test image:
# test_image = Image.open("/home/al0nkr/visual-search-rag/calib_imgs/000010004.jpg")  # or Image.open("some_image.jpg").convert("RGB")

# # 2) Build a chat-style prompt, like in your calibration step
# messages = [
#     {"role": "user", "content": "<|image_1|>\nWhat does this image show?"},
# ]

# prompt = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,   # we want the model to answer now
#     tokenize=False,
# )

# # 3) Use the processor to create ALL inputs (text + image)
# inputs = processor(
#     text=messages,
#     images=test_image,
#     return_tensors="pt",
# )

# # Move to the model device
# inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
#           for k, v in inputs.items()}

# # 4) Generate — IMPORTANT: pass the whole dict with **inputs
# with torch.inference_mode():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=128,
#         do_sample=False,  # greedy
#     )

# # 5) Decode
# decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
# print(decoded)
# print("==========================================")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)