# Phase 7a: Tokenizer

## Goal
Build the tokenizer that converts expression trees to/from integer sequences.
Used by BOTH architectures (sequence transformer and tree GNN).

## What a Tokenizer Does
Models can't read math symbols. They read numbers.
The tokenizer is a dictionary: "sin" → 20, "x" → 70, "add" → 10.
It converts `sin(x+y)` → the tree `sin(add(x, y))` → the sequence `[20, 10, 70, 71]`.

## File: `python/neurips/data/tokenizer.py`

### Token Vocabulary (~256 tokens)
```python
VOCAB = {
    # Special tokens (IDs 0-9)
    "<PAD>": 0,    # padding — makes sequences the same length in a batch
    "<SOS>": 1,    # start-of-sequence — tells decoder to start generating
    "<EOS>": 2,    # end-of-sequence — tells decoder to stop
    "<UNK>": 3,    # unknown — safety fallback for unexpected symbols
    "[FEAT]": 4,   # feature injection — carries structural info into the model
    "[INDEF]": 5,  # task token: indefinite integral
    "[DEF]": 6,    # task token: definite integral
    "[VAR]": 7,    # next token is the integration variable
    "[PARAM]": 8,  # next token is a symbolic parameter
    "[LOWER]": 9,  # next token is lower bound of definite integral

    # Binary operators (IDs 10-15)
    "add": 10, "sub": 11, "mul": 12, "div": 13, "pow": 14, "neg": 15,

    # Elementary functions (IDs 20-31)
    "sin": 20, "cos": 21, "tan": 22, "exp": 23, "log": 24, "sqrt": 25,
    "asin": 26, "acos": 27, "atan": 28, "sinh": 29, "cosh": 30, "tanh": 31,

    # Special functions (IDs 40-54)
    "erf": 40, "Ei": 41, "Si": 42, "Ci": 43, "Li": 44,
    "BesselJ": 45, "BesselY": 46, "EllipticK": 47, "EllipticE": 48,
    "Gamma": 49, "digamma": 50, "FresnelS": 51, "FresnelC": 52,
    "Hyp2F1": 53, "polylog": 54,

    # Variables (IDs 70-74)
    "x": 70, "y": 71, "z": 72, "w": 73, "t": 74,

    # Parameters (IDs 80-86)
    "a": 80, "b": 81, "c": 82, "n": 83, "k": 84, "alpha": 85, "beta": 86,

    # Constants (IDs 90-93)
    "pi": 90, "euler_gamma": 91, "catalan": 92, "inf": 93,

    # Number encoding (IDs 100-201): base-100 system
    "+": 100, "-": 101,   # sign tokens
    # IDs 102-201: digits 0-99 (encode two decimal digits per token)
}
```

### Base-100 Number Encoding
The number 347 is encoded as: sign(+), digit(3), digit(47) → [100, 105, 149].
Why? If you encode each digit separately (0-9), 347 is 3 tokens.
Base-100: 9999 = [100, 201, 201] = 3 tokens instead of 4.
Saves ~20% token length for typical expressions.

### Tokenizer Class
```python
class IntegralTokenizer:
    def encode(self, tree, task, var, bounds=None, params=None) -> list[int]:
        """Tree + metadata → token IDs.
        Format: [task] [VAR] var [FEAT] [prefix tokens...] [EOS]
        """
    def decode(self, token_ids: list[int]) -> tuple[ExprNode, dict]:
        """Token IDs → tree + metadata."""
    def vocab_size(self) -> int:
        return 256
```

## Verification
- Round-trip: `decode(encode(tree)) == tree` for 10K random trees
- All task types encode/decode correctly
- Base-100: encode(347) → [100, 105, 149], decode back → 347
- Negative numbers: encode(-42) → [101, 144]
- Max sequence length never exceeds 512 tokens for generated data
