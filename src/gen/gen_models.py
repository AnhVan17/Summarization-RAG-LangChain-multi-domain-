from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

GEN_MODEL = "VietAI/vit5-base-vietnews-summarization"
tok = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_text(prompt, max_new=220, min_new=80):
    inp = tok(prompt, return_tensors="pt", truncation=True,
              max_length=1024, padding=True).to(device)
    out = model.generate(
        **inp, max_new_tokens=max_new, min_new_tokens=min_new,
        num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.15,
        length_penalty=1.2, early_stopping=False
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return re.sub(r"<extra_id_\d+>", "", text).strip()
