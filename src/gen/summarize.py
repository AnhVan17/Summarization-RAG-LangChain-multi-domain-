from .gen_models import generate_text


def summarize_tldr(doc, sentences=5):
    p = f"Hãy tóm tắt văn bản sau thành khoảng {sentences} câu, mạch lạc, đúng ý gốc:\n\n{doc}\n\nTÓM TẮT:"
    return generate_text(p)


def summarize_exec(doc, points=6):
    p = f"Hãy tóm tắt văn bản sau thành {points} ý chính, xuất dạng bullet '- ':\n\n{doc}\n\nTÓM TẮT:"
    out = generate_text(p)
    return out if out.strip().startswith("-") else "- " + out.replace("\n", "\n- ")


def map_reduce_summary(chunks, reduce_points=7):
    partials = [generate_text(
        f"Tóm tắt đoạn sau thành 2 câu:\n\n{c}\n\nTÓM TẮT:", max_new=140, min_new=50) for c in chunks]
    corpus = "\n".join(partials)
    p = (f"Gộp các ý sau thành {reduce_points} ý chính, không trùng lặp, dạng bullet '- ':\n\n"
         f"{corpus}\n\nTÓM TẮT CUỐI:")
    return generate_text(p, max_new=300, min_new=120)
