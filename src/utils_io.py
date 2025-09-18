from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import re
from src.preprocess import TextPreprocessor

_PRE = TextPreprocessor(language="vi")


def _norm(text: str) -> str:
    return _PRE.preprocess(text)


def load_pdf(path: str) -> str:
    """Đọc PDF bằng PyMuPDF và trả về text ghép của tất cả trang."""
    try:
        import fitz
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF (fitz) chưa được cài. Cài bằng: pip install pymupdf"
        ) from e

    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return _norm("\n".join(text_parts))


def _safe_read_text(p: Path, encoding: str) -> Optional[str]:
    try:
        return p.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return None


def load_txt(path: Union[str, Path]) -> str:
    """Đọc TXT với fallback encoding phổ biến (ưu tiên UTF-8)."""
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig", "cp1258", "utf-16", "latin-1"):
        s = _safe_read_text(p, enc)
        if s is not None:
            return _norm(s)
    return _norm(p.read_text(encoding="utf-8", errors="ignore"))


def load_file(path: Union[str, Path]) -> str:
    """Chọn loader theo đuôi file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(str(path))
    return load_txt(path)


def write_text(path: Union[str, Path], content: str) -> None:
    """Ghi file an toàn (tạo thư mục nếu thiếu)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)
