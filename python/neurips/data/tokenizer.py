"""Tokenizer: convert prefix expression strings to/from integer sequences."""

from __future__ import annotations

from neurips.data.prefix import tokenize_prefix
from neurips.data.vocab import (
    EOS,
    FEAT,
    ID_TO_TOKEN,
    PARAM,
    TASK_TOKENS,
    UNK,
    VAR,
    VOCAB,
    VOCAB_SIZE,
)

_SIGN_PLUS: int = VOCAB["+"]
_SIGN_MINUS: int = VOCAB["-"]
_DIGIT_BASE: int = 102  # digit 0 -> ID 102, digit 99 -> ID 201

# Structural token IDs to skip when decoding prefix content
_META_IDS: frozenset[int] = frozenset({
    VOCAB["<PAD>"], VOCAB["<SOS>"], VOCAB["<EOS>"],
    VOCAB["<UNK>"], FEAT, VOCAB["[INDEF]"], VOCAB["[DEF]"],
    VAR, PARAM, VOCAB["[LOWER]"],
})


class IntegralTokenizer:
    """Encode/decode prefix-notation integral expressions as token IDs."""

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    # ── Number encoding ─────────────────────────────────────────────

    def encode_number(self, n: int) -> list[int]:
        """Encode integer *n* in base-100 with a sign token prefix."""
        sign_tok = _SIGN_PLUS if n >= 0 else _SIGN_MINUS
        n = abs(n)
        if n == 0:
            return [sign_tok, _DIGIT_BASE]
        digits: list[int] = []
        while n > 0:
            digits.append(_DIGIT_BASE + (n % 100))
            n //= 100
        digits.reverse()
        return [sign_tok, *digits]

    def decode_number(self, tokens: list[int]) -> int:
        """Decode a base-100 encoded number from token IDs."""
        if not tokens:
            raise ValueError("Empty token list for number decoding")
        sign = -1 if tokens[0] == _SIGN_MINUS else 1
        value = 0
        for tid in tokens[1:]:
            digit = tid - _DIGIT_BASE
            if not (0 <= digit <= 99):
                raise ValueError(f"Invalid digit token ID: {tid}")
            value = value * 100 + digit
        return sign * value

    # ── Full sequence encoding ──────────────────────────────────────

    def encode(
        self,
        prefix_str: str,
        task: str,
        var: str,
        bounds: tuple[str, str] | None = None,
        params: list[str] | None = None,
    ) -> list[int]:
        """Encode a prefix expression with metadata into token IDs.

        Format: [task] [VAR] var [PARAM params...] [FEAT] [prefix...] [EOS]
        """
        ids: list[int] = []
        task_id = TASK_TOKENS.get(task.lower())
        if task_id is None:
            raise ValueError(f"Unknown task type: {task}")
        ids.append(task_id)

        ids.append(VAR)
        ids.append(VOCAB.get(var, UNK))

        if params:
            ids.append(PARAM)
            for p in params:
                ids.append(VOCAB.get(p, UNK))

        ids.append(FEAT)

        for tok in tokenize_prefix(prefix_str):
            ids.append(VOCAB.get(tok, UNK))

        ids.append(EOS)
        return ids

    # ── Full sequence decoding ──────────────────────────────────────

    def decode(self, token_ids: list[int]) -> tuple[str, dict]:
        """Decode token IDs back to prefix string + metadata dict."""
        meta: dict = {"task": None, "var": None, "params": []}
        prefix_tokens: list[str] = []
        section = "task"  # expect task token first

        for tid in token_ids:
            if tid == EOS:
                break
            if tid in (VOCAB["<PAD>"], VOCAB["<SOS>"]):
                continue

            name = ID_TO_TOKEN.get(tid, "<UNK>")

            if section == "task":
                # First meaningful token is the task
                for k, v in TASK_TOKENS.items():
                    if v == tid:
                        meta["task"] = k
                        break
                section = "header"
            elif tid == VAR and section == "header":
                section = "var"
            elif section == "var":
                meta["var"] = name
                section = "header"
            elif tid == PARAM:
                section = "param"
            elif section == "param" and tid != FEAT:
                meta["params"].append(name)
            elif tid == FEAT:
                section = "prefix"
            elif section == "prefix":
                prefix_tokens.append(name)

        return " ".join(prefix_tokens), meta
