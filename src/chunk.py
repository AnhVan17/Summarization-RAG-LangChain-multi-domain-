import re


def chunk_by_words(text: str, chunk_size=220, overlap=40):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks


def sent_split(text: str):
    parts = re.split(r'(?<=[\\.!?…])\\s+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_sentences(text: str, max_chars=900, overlap_sents=1):
    sents = sent_split(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents > 0 else []
            cur_len = sum(len(x)+1 for x in cur)
        cur.append(s)
        cur_len += len(s)+1
    if cur:
        chunks.append(" ".join(cur))
    return chunks
