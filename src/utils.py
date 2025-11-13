import re
import html
import json
from collections import Counter
from typing import List, Dict, Any

import numpy as np


# ----------------------------
# 1) Text cleaning
# ----------------------------

# IMDb reviews often contain <br /> and HTML entities, so we handle those.
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Keep only lowercase letters, digits, whitespace. Everything else -> space.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    """
    Clean raw review text according to the project specification:

    1. Lowercase all text.
    2. Remove punctuation and special characters.
    3. (Additionally) remove HTML tags and URLs.
    4. Collapse multiple spaces.

    Parameters
    ----------
    text : str
        Raw review text from the IMDb dataset.

    Returns
    -------
    str
        Cleaned, tokenization-ready string.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Lowercase
    text = text.lower()

    # Unescape HTML entities: &amp; -> &, etc.
    text = html.unescape(text)

    # Remove URLs
    text = _URL_RE.sub(" ", text)

    # Remove HTML tags such as <br />
    text = _TAG_RE.sub(" ", text)

    # Remove punctuation and special characters
    text = _NON_ALNUM_RE.sub(" ", text)

    # Collapse multiple spaces and strip leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------
# 2) Tokenizer
# ----------------------------

class LiteTokenizer:
    """
    A lightweight tokenizer with Keras-like behavior, but no external dependencies.

    Design:
    - We assume input texts are already cleaned and tokenized by whitespace.
    - PAD token index = 0, OOV (out-of-vocabulary) token index = 1.
    - We keep at most (num_words - 2) actual vocabulary tokens, since we
      reserve 0 and 1 for PAD and OOV.

    Example
    -------
    tok = LiteTokenizer(num_words=10_000, oov_token="<OOV>")
    tok.fit_on_texts(list_of_clean_strings)
    seqs = tok.texts_to_sequences(list_of_clean_strings)
    """

    def __init__(self, num_words: int = 10_000, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token

        self.word_counts: Counter = Counter()
        self.word_index: Dict[str, int] = {}   # word -> idx
        self.index_word: Dict[int, str] = {}   # idx -> word

        # We will always treat:
        #   0: PAD
        #   1: OOV
        self.PAD_IDX = 0
        self.OOV_IDX = 1

    # --------- fitting ---------

    def fit_on_texts(self, texts: List[str]) -> None:
        """
        Build the vocabulary based on list of cleaned texts (space-separated tokens).

        We count token frequencies and keep the most frequent tokens,
        respecting the num_words constraint.
        """
        # Count token frequencies
        for t in texts:
            if not isinstance(t, str):
                continue
            for tok in t.split():
                self.word_counts[tok] += 1

        # How many "real" tokens can we keep?
        # We reserve index 0 for PAD and 1 for OOV, so we have (num_words - 2)
        # slots for actual words.
        max_vocab_size = max(self.num_words - 2, 0)

        # most_common returns list of (token, count) sorted descending by count
        most_common = self.word_counts.most_common(max_vocab_size)

        # Build word -> index mapping
        self.word_index = {}
        self.index_word = {
            self.PAD_IDX: "<PAD>",
            self.OOV_IDX: self.oov_token,
        }

        current_idx = 2  # start assigning indices from 2
        for word, _freq in most_common:
            self.word_index[word] = current_idx
            self.index_word[current_idx] = word
            current_idx += 1

    # --------- transformation ---------

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert each cleaned text string into a list of integer token IDs.

        Unknown tokens map to OOV index.
        """
        sequences = []
        for t in texts:
            if not isinstance(t, str):
                t = "" if t is None else str(t)
            tokens = t.split()
            ids = [self.word_index.get(tok, self.OOV_IDX) for tok in tokens]
            sequences.append(ids)
        return sequences

    # --------- serialization ---------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "num_words": self.num_words,
            "oov_token": self.oov_token,
            "PAD_IDX": self.PAD_IDX,
            "OOV_IDX": self.OOV_IDX,
            "word_index": self.word_index,
            "index_word": self.index_word,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiteTokenizer":
        """Create a LiteTokenizer from a dictionary (e.g., loaded from JSON)."""
        obj = cls(num_words=data["num_words"], oov_token=data["oov_token"])
        obj.PAD_IDX = data.get("PAD_IDX", 0)
        obj.OOV_IDX = data.get("OOV_IDX", 1)
        obj.word_index = data["word_index"]
        obj.index_word = {int(k): v for k, v in data["index_word"].items()}
        return obj

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "LiteTokenizer":
        """Deserialize from JSON string."""
        data = json.loads(s)
        return cls.from_dict(data)


# ----------------------------
# 3) Padding
# ----------------------------

def pad_sequences(
    sequences: List[List[int]],
    maxlen: int,
    padding: str = "post",
    truncating: str = "post",
    value: int = 0,
) -> np.ndarray:
    """
    Pad or truncate a list of sequences to a fixed length.

    This mirrors the behavior of keras.preprocessing.sequence.pad_sequences,
    which is what many RNN-based sentiment models expect:

    - truncating="post": keep the first maxlen tokens.
    - padding="post": pad at the end with `value` until length=maxlen.

    Parameters
    ----------
    sequences : list of list of int
        Each inner list is a token ID sequence.
    maxlen : int
        Desired fixed length for each sequence.
    padding : {"pre", "post"}
        Where to add padding tokens (default "post").
    truncating : {"pre", "post"}
        Which part of the sequence to truncate (default "post").
    value : int
        Pad value (should match tokenizer.PAD_IDX).

    Returns
    -------
    np.ndarray of shape (num_sequences, maxlen), dtype=int32
    """
    n = len(sequences)
    out = np.full((n, maxlen), value, dtype=np.int32)

    for i, seq in enumerate(sequences):
        if not seq:
            continue

        # Truncate
        if len(seq) > maxlen:
            if truncating == "pre":
                seq = seq[-maxlen:]
            else:  # "post"
                seq = seq[:maxlen]

        # Pad
        if padding == "pre":
            out[i, -len(seq):] = seq
        else:  # "post"
            out[i, :len(seq)] = seq

    return out