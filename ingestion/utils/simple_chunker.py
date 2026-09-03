from typing import List

from transformers import AutoTokenizer


class SimpleChunker:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 300,
    ):
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """A paragraph longer than max_tokens would be silently truncated by
        the embedding model, so split it on the token budget."""
        token_ids = self.tokenizer.encode(paragraph, add_special_tokens=False)

        if len(token_ids) <= self.max_tokens:
            return [paragraph]

        return [
            self.tokenizer.decode(token_ids[i : i + self.max_tokens])
            for i in range(0, len(token_ids), self.max_tokens)
        ]

    def create_chunks(self, text_content: str):
        paragraphs = [p.strip() for p in text_content.split("\n") if p.strip()]
        paragraphs = [
            piece for p in paragraphs for piece in self._split_long_paragraph(p)
        ]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para, add_special_tokens=False))

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
