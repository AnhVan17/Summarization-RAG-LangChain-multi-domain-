import fitz  # PyMuPDF
import re
from pathlib import Path


def load_pdf(path: str) -> str:
    """Đọc PDF bằng PyMuPDF và trả về text ghép của tất cả trang."""
    text_parts = []
    # Dùng context manager để tự đóng file
    with fitz.open(path) as doc:
        for page in doc:  # page là đối tượng Page
            # "text" = plain text; có thể đổi "blocks" nếu muốn giữ layout cơ bản
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def load_txt(path: str) -> str:
    # Thử utf-8 trước, nếu lỗi thì fallback
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return p.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return p.read_text(encoding="cp1258", errors="ignore")


def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()
