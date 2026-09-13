"""Tokenizers. Default is byte-level — zero dependencies, zero out-of-vocab.

byte: every utf-8 byte is a token, vocab is exactly 256. Trivial, universal,
and honest about what a tiny model can learn from a small dataset.

bpe: imports a tokenizer already trained by nanobrain (sentencepiece .model,
remapped to the nanollama.c id convention: 0/1/2 specials, bytes at 3..258).
Requires the optional `sentencepiece` package.
"""

from __future__ import annotations


class ByteTokenizer:
    """Text <-> ids in the nanollama.c convention: 0/1/2 = <unk>/<s>/</s>,
    every utf-8 byte at id 3+b (the engine's hardcoded byte-fallback rule).

    This makes byte models engine-ready by construction: the C engine can
    decode their output with any llama2-style tokenizer file, no merging
    vocabulary required. Vocab is 259 (specials + 256 bytes)."""

    UNK, BOS, EOS = 0, 1, 2
    BYTE_OFFSET = 3
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return [self.BYTE_OFFSET + b for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if self.BYTE_OFFSET <= i < self.BYTE_OFFSET + 256:
                out.append(i - self.BYTE_OFFSET)
        return out.decode("utf-8", errors="replace")

    @property
    def kind(self) -> str:
        return "byte"


class BpeTokenizer:
    """Loads a nanobrain-trained tokenizer and remaps ids to the engine
    convention (0/1/2 = <unk>/<s>/</s>, raw bytes at 3..258) — same ID_MAP
    dance as nanobrain/nanobrain/tokenizer.py, so exported models line up
    with nanollama.c's hardcoded byte-fallback."""

    UNK, BOS, EOS, N_BYTES = 0, 1, 2, 256

    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "tokenizer: bpe needs the optional `sentencepiece` package "
                "(pip install sentencepiece), or switch to tokenizer: byte"
            ) from e
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self._build_id_map()

    def _build_id_map(self) -> None:
        sp_to_final: dict[int, int] = {}
        final_to_sp: list[int] = [self.UNK, self.BOS, self.EOS]
        sp_to_final[self.UNK], sp_to_final[self.BOS], sp_to_final[self.EOS] = 0, 1, 2
        for sp_id in range(3, self.sp.get_piece_size()):
            piece = self.sp.id_to_piece(sp_id)
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    byte_val = int(piece[3:5], 16)
                except ValueError:
                    continue
                if 0 <= byte_val < self.N_BYTES:
                    final = 3 + byte_val
                    sp_to_final[sp_id] = final
                    while len(final_to_sp) <= final:
                        final_to_sp.append(-1)
                    final_to_sp[final] = sp_id
        for sp_id in range(3, self.sp.get_piece_size()):
            if sp_id not in sp_to_final:
                final_to_sp.append(sp_id)
                sp_to_final[sp_id] = len(final_to_sp) - 1
        self.sp_to_final = sp_to_final
        self.final_to_sp = final_to_sp

    @property
    def kind(self) -> str:
        return "bpe"

    @property
    def vocab_size(self) -> int:
        return len(self.final_to_sp)

    def encode(self, text: str) -> list[int]:
        return [self.sp_to_final[i] for i in self.sp.encode(text)]

    def decode(self, ids: list[int]) -> str:
        sp_ids = [self.final_to_sp[i] for i in ids if 0 <= i < len(self.final_to_sp)]
        return self.sp.decode(sp_ids)


def make_tokenizer(kind: str, model_path: str | None = None):
    if kind == "byte":
        return ByteTokenizer()
    if kind == "bpe":
        if not model_path:
            raise ValueError("tokenizer: bpe requires `bpe_model` in the spec")
        return BpeTokenizer(model_path)
    raise ValueError(f"unknown tokenizer '{kind}'")
