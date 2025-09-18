import re
import unicodedata
import string
from typing import Optional, List, Any, Dict, Callable

_HTML = re.compile(r"<[^>]+>")
_MULTISPACE = re.compile(r"\s+")
_HYPHEN_LINE = re.compile(r"(\w)-\n(\w)")


class TextPreprocessor:
    def __init__(self, language: str = 'vi', custom_stopwords: Optional[List[str]] = None):
        self.language = language
        self.custom_stopwords = set(custom_stopwords or [])
        self.punctuation = set(string.punctuation)

    def _safe_text(self, text: Any) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        return text.strip()

    def normalize_newlines(self, text: str) -> str:
        t = self._safe_text(text)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", self._safe_text(text))

    def basic_cleaning(self, text: str) -> str:
        t = self._safe_text(text)
        t = _HYPHEN_LINE.sub(r"\1\2", t)
        t = _HTML.sub(" ", t)
        t = _MULTISPACE.sub(" ", t).strip()
        return t

    def collapse_spaces(self, text: str) -> str:
        return _MULTISPACE.sub(" ", self._safe_text(text)).strip()

    def remove_punctuation(self, text: str, keep_sentence_ending: bool = True) -> str:
        t = self._safe_text(text)
        rm = self.punctuation - \
            {'.', '!', '?'} if keep_sentence_ending else self.punctuation
        t = ''.join(ch if ch not in rm else ' ' for ch in t)
        return _MULTISPACE.sub(" ", t).strip()

    def lowercase(self, text: str) -> str:
        return self._safe_text(text).lower()

    def preprocess(self, text: Any, steps: Optional[List[str]] = None, verbose: bool = False, **kwargs) -> str:
        t = self._safe_text(text)
        if not t:
            if verbose:
                print("[WARN] empty input")
            return ""

        pipelines: Dict[str, Callable[[str], str]] = {
            "normalize_newlines": self.normalize_newlines,
            "normalize_unicode": self.normalize_unicode,
            "basic_cleaning": self.basic_cleaning,
            "collapse_spaces": self.collapse_spaces,

            "remove_punctuation": lambda x: self.remove_punctuation(x, kwargs.get("keep_sentence_ending", True)),
            "lowercase": self.lowercase,
        }

        steps = steps or ["normalize_newlines",
                          "normalize_unicode", "basic_cleaning", "collapse_spaces"]

        if verbose:
            print(f"[INFO] Original: {repr(t)}")
        for s in steps:
            f = pipelines.get(s)
            if f:
                t = f(t)
                if verbose:
                    print(f"[{s}] -> {repr(t)}")
                if not t.strip():
                    if verbose:
                        print(f"[WARN] empty after {s}")
                    break
        return self.collapse_spaces(t)
