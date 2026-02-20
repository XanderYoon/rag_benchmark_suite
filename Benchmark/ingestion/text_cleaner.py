from __future__ import annotations

import re


class TextCleaner:
    def clean(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        # Remove surrogate code points emitted by some PDF extractors. These
        # are invalid in UTF-8 and will fail when writing text to disk.
        text = re.sub(r"[\ud800-\udfff]", "", text)
        return text.strip()
