from __future__ import annotations
from typing import Literal, Dict, Any
import re

Mode = Literal["tldr", "executive", "qfs"]


def detect_lang_fast(s: str) -> str:
    # heuristic nhẹ: có dấu tiếng Việt -> 'vi', ngược lại 'en'
    return "vi" if re.search(r"[àáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ]", s.lower()) else "en"


# Prompt templates (ngắn gọn, có guardrail & citation rule)
PROMPT_VI = {
    "tldr": """Bạn là trợ lý súc tích.
CHỈ dùng THÔNG TIN TỪ NGỮ CẢNH dưới đây (có [n]).
Nếu thiếu thông tin → trả: "Không đủ thông tin".

YÊU CẦU: Tóm tắt TL;DR gọn trong 2–4 câu, tiếng Việt, không bịa, có thể chèn [n] sau câu phù hợp.

NGỮ CẢNH:
{context}

KẾT QUẢ:
""",
    "executive": """Bạn là trợ lý cho lãnh đạo.
CHỈ dùng THÔNG TIN TỪ NGỮ CẢNH (có [n]); thiếu → "Không đủ thông tin".
Xuất bullet Executive (4–6 gạch đầu dòng), tiếng Việt, không bịa. Có thể chèn [n].

NGỮ CẢNH:
{context}

KẾT QUẢ:
""",
    "qfs": """Bạn là trợ lý trả lời theo truy vấn.
CHỈ dùng NGỮ CẢNH (có [n]); thiếu → "Không đủ thông tin".
Trả lời trực tiếp, ngắn gọn, tiếng Việt; có thể chèn [n] sau mệnh đề có chứng cứ.

CÂU HỎI: {question}

NGỮ CẢNH:
{context}

KẾT QUẢ:
""",
}

PROMPT_EN = {
    "tldr": """You are a concise assistant.
Answer ONLY from the CONTEXT below (with [n]); if missing → "Insufficient context".
Return a TL;DR in 2–4 sentences, in English. You may insert [n] where justified.

CONTEXT:
{context}

ANSWER:
""",
    "executive": """You assist executives.
Answer ONLY from CONTEXT; if missing → "Insufficient context".
Return 4–6 executive bullets, in English. You may place [n] after supported claims.

CONTEXT:
{context}

ANSWER:
""",
    "qfs": """Answer only from CONTEXT; if missing → "Insufficient context".
Give a direct, concise answer in English. You may insert [n] after supported statements.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
""",
}


def build_prompt(context_annotated: str, question: str, mode: Mode, out_lang: str) -> str:
    out_lang = (out_lang or "en").lower()
    tpl = PROMPT_VI if out_lang == "vi" else PROMPT_EN
    return tpl[mode].format(context=context_annotated.strip(), question=question.strip())
