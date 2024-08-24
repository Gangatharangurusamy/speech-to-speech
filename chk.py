import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

device = "cpu"
try:
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load model or tokenizer: {e}")
