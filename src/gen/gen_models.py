from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import os


os.environ.setdefault("TRANSFORMERS_CACHE", "./.cache/transformers")
GEN_MODEL = "VietAI/vit5-base-vietnews-summarization"

tok = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

_CLEAN_EXTRA_ID = re.compile(r"<extra_id_\d+>")


@torch.inference_mode()
def generate_text(prompt: str, max_new=240, min_new=90):
    """Sinh văn bản mạch lạc, tránh cụt/lặp."""
    inp = tok(prompt, return_tensors="pt",
              truncation=True, max_length=1024).to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        min_new_tokens=min_new,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=1.2,
        early_stopping=True,
        pad_token_id=tok.pad_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return _CLEAN_EXTRA_ID.sub("", text).strip()


def estimate_tokens(text: str) -> int:
    return len(tok(text, truncation=False, add_special_tokens=False)["input_ids"])
