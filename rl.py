import re
import json
import difflib
import argparse
from typing import Optional, List, Tuple

import torch
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2TokenizerFast,
)
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, ".")
from voter_profiles import build_voter_profile, ALL_QUESTIONS

nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("words", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import words as _nltk_words
from nltk.corpus import wordnet as _wn
_ENGLISH_WORDS = set(w.lower() for w in _nltk_words.words())

try:
    from langdetect import detect as _langdetect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed. English constraint disabled. pip install langdetect")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# SYSTEM_PROMPT = "You are a precise text classifier. Answer only 'yes' or 'no'."
# QUESTION      = "Does this text contain fancy or sophisticated vocabulary?"

PROP_ANCHOR   = "=== BALLOT PROPOSITION ===\n"
SYSTEM_PROMPT = (
    "You are simulating a voter deciding how to vote on a ballot proposition. "
    "Read the voter profile and the proposition, then respond with a JSON object "
    'containing your vote: {"vote": "YES"} or {"vote": "NO"}. '
    "Base your vote on the voter's profile and values."
)
VOTER_PROFILE = "" 

_qwen_tok:   Optional[AutoTokenizer]        = None
_qwen_model: Optional[AutoModelForCausalLM] = None
_yes_id:     Optional[int] = None
_no_id:      Optional[int] = None


def _init_reward_model():
    global _qwen_tok, _qwen_model, _yes_id, _no_id
    if _qwen_tok is not None:
        return
    print("Loading Qwen2.5-7B-Instruct (reward model)…")
    _qwen_tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    _qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    _yes_id = _qwen_tok("YES", add_special_tokens=False)["input_ids"][0]
    _no_id  = _qwen_tok("NO",  add_special_tokens=False)["input_ids"][0]



def _build_prompt_ids(text: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": VOTER_PROFILE + "\n\n" + PROP_ANCHOR + text},
    ]
    prompt = _qwen_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + '{"vote": "'
    return _qwen_tok(prompt, return_tensors="pt").input_ids.to(DEVICE)


def reward(text: str, with_grad: bool = False):
    """
    Log-odds of "yes" over "no": logit_yes - logit_no.
    Positive = model leans yes, negative = model leans no.
    Unbounded — never saturates regardless of model confidence.

    Gradient flows through the subtraction directly.

    with_grad=False  ->  (float, None, None)
    with_grad=True   ->  (float, grad, ids)
                           grad : (seq_len, hidden)  d(log_odds)/d(embeddings)
                           ids  : (1, seq_len)
    """
    ids = _build_prompt_ids(text)

    if not with_grad:
        with torch.no_grad():
            logits   = _qwen_model(ids).logits[0, -1].float()
            log_odds = logits[_yes_id] - logits[_no_id]
            return log_odds.item(), None, None

    emb_layer  = _qwen_model.model.embed_tokens
    embeddings = emb_layer(ids).detach().requires_grad_(True)
    logits     = _qwen_model(inputs_embeds=embeddings).logits[0, -1].float()
    log_odds   = logits[_yes_id] - logits[_no_id]
    log_odds.backward()

    return log_odds.item(), embeddings.grad[0].detach(), ids



_embedder: Optional[SentenceTransformer] = None


def _init_embedder():
    global _embedder
    if _embedder is not None:
        return
    print("Loading mxbai-embed-large-v1 (similarity)…")
    _embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=DEVICE)


def _embed(texts: List[str]) -> np.ndarray:
    return _embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _sim(a: str, b: str) -> float:
    v = _embed([a, b])
    return float(v[0] @ v[1])



_POS_GROUP = {
    # Nouns (coarse — singular/plural can swap)
    "NN":   "NOUN",   "NNS":  "NOUN",
    # Proper nouns — block (names shouldn't be swapped)
    "NNP":  "PROPN",  "NNPS": "PROPN",
    # Verbs — fine-grained so tense must match
    "VB":   "VB",     # base form
    "VBD":  "VBD",    # past tense
    "VBG":  "VBG",    # present participle / gerund
    "VBN":  "VBN",    # past participle
    "VBP":  "VBP",    # non-3rd-person singular present
    "VBZ":  "VBZ",    # 3rd-person singular present
    # Adjectives (coarse)
    "JJ":   "ADJ",    "JJR":  "ADJ",   "JJS":  "ADJ",
    # Adverbs (coarse)
    "RB":   "ADV",    "RBR":  "ADV",   "RBS":  "ADV",
    # Everything below is blocked from substitution
    "CD":   "NUM",
    "PRP":  "PRON",   "PRP$": "PRON",  "WP":  "PRON",  "WP$": "PRON",
    "DT":   "DET",    "IN":   "PREP",  "TO":  "PREP",
    "CC":   "CONJ",
    "WDT":  "WH",     "WRB":  "WH",
    "MD":   "MODAL",  # modals (can/will/should) — block
    "RP":   "PART",   # particles — block
    "EX":   "EXIST",  # existential there — block
    "UH":   "INTJ",   # interjections — block
}

# Tags that are in the map but should never be swapped
_BLOCKED_GROUPS = {"PROPN", "NUM", "PRON", "DET", "PREP", "CONJ",
                   "WH", "MODAL", "PART", "EXIST", "INTJ"}


def _fine_pos(word: str, sentence: str) -> Optional[str]:
    """Return the fine-grained POS group for `word` in `sentence`, or None."""
    word_clean = re.sub(r"[^\w]", "", word).lower()
    tokens = word_tokenize(sentence)
    tags   = pos_tag(tokens)
    for tok, tag in tags:
        if re.sub(r"[^\w]", "", tok).lower() == word_clean:
            return _POS_GROUP.get(tag, "UNKNOWN")
    return None


def _pos_ok(orig_word: str, sub_word: str, orig_sentence: str, trial_sentence: str) -> bool:
    """True only if sub_word has exactly the same POS group as orig_word,
    and neither is in a blocked group. Strict — unknown tags are blocked.
    """
    orig_pos = _fine_pos(orig_word, orig_sentence)
    if orig_pos is None or orig_pos in _BLOCKED_GROUPS | {"UNKNOWN"}:
        return False
    sub_pos = _fine_pos(sub_word, trial_sentence)
    if sub_pos is None or sub_pos in _BLOCKED_GROUPS | {"UNKNOWN"}:
        return False
    return orig_pos == sub_pos

_gpt2_tok:   Optional[GPT2TokenizerFast] = None
_gpt2_model: Optional[GPT2LMHeadModel]  = None


def _init_gpt2():
    global _gpt2_tok, _gpt2_model
    if _gpt2_tok is not None:
        return
    print("Loading GPT-2 (perplexity filter)…")
    _gpt2_tok              = GPT2TokenizerFast.from_pretrained("gpt2")
    _gpt2_tok.pad_token    = _gpt2_tok.eos_token
    _gpt2_model            = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()


def _perplexity(text: str) -> float:
    """Whole-text GPT-2 perplexity."""
    enc = _gpt2_tok(text, return_tensors="pt", truncation=True,
                    max_length=512).to(DEVICE)
    with torch.no_grad():
        loss = _gpt2_model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()


def _local_perplexity(text: str, word: str, window: int = 5) -> float:
    """
    GPT-2 perplexity on a small window around `word` in `text`.
    Extracts up to `window` words before and after the first occurrence
    of `word`, then scores just that fragment.
    This catches local grammatical issues..
    """
    words  = text.split()
    target = re.sub(r"[^\w]", "", word).lower()
    idx    = next((i for i, w in enumerate(words)
                   if re.sub(r"[^\w]", "", w).lower() == target), None)
    if idx is None:
        return _perplexity(text)  
    start   = max(0, idx - window)
    end     = min(len(words), idx + window + 1)
    snippet = " ".join(words[start:end])
    return _perplexity(snippet)


def _ppl_ok(trial_text: str, sub_word: str, orig_ppl: float,
            orig_local_ppl: float, ppl_factor: float) -> bool:
    trial_ppl = _perplexity(trial_text)
    if trial_ppl > ppl_factor * orig_ppl:
        return False, trial_ppl
    trial_local = _local_perplexity(trial_text, sub_word)
    if trial_local > ppl_factor * orig_local_ppl:
        return False, trial_ppl
    return True, trial_ppl



def _wordnet_pos(word: str) -> Optional[str]:
    """
    Return the most common WordNet POS for `word` ('n','v','a','r'), or None.
    Uses the first (most frequent) synset.
    """
    synsets = _wn.synsets(word.lower())
    if not synsets:
        return None
    return synsets[0].pos()   # 'n', 'v', 'a', 'r', 's'


def _wordnet_ok(orig_word: str, sub_word: str) -> bool:
    """
    True if orig_word and sub_word share the same WordNet POS category.
    'a' and 's' (adjective satellite) are treated as the same.
    Returns False if either word has no WordNet entry — unknown words blocked.
    """
    orig_wpos = _wordnet_pos(orig_word)
    sub_wpos  = _wordnet_pos(sub_word)
    if orig_wpos is None or sub_wpos is None:
        return False   # not in WordNet → block
    orig_wpos = "a" if orig_wpos == "s" else orig_wpos
    sub_wpos  = "a" if sub_wpos  == "s" else sub_wpos
    return orig_wpos == sub_wpos

def _inflect(orig_word: str, sub_word: str, orig_sentence: str) -> str:
    """
    Return sub_word inflected to match the morphological form of orig_word.
    """
    word_clean = re.sub(r"[^\w]", "", orig_word).lower()
    tokens     = word_tokenize(orig_sentence)
    tags       = pos_tag(tokens)
    orig_tag   = next(
        (tag for tok, tag in tags
         if re.sub(r"[^\w]", "", tok).lower() == word_clean),
        None
    )

    result = sub_word.lower()

    if orig_tag == "NNS":                         # plural noun
        if not result.endswith("s"):
            result = result + "es" if result.endswith(("s","sh","ch","x","z")) else result + "s"
    elif orig_tag == "VBZ":                       # 3rd person singular present
        if not result.endswith("s"):
            result = result + "es" if result.endswith(("s","sh","ch","x","z")) else result + "s"
    elif orig_tag == "VBD" or orig_tag == "VBN":  # past tense / past participle
        if not result.endswith("ed"):
            result = result[:-1] + "ed" if result.endswith("e") else result + "ed"
    elif orig_tag == "VBG":                       # present participle
        if not result.endswith("ing"):
            result = result[:-1] + "ing" if result.endswith("e") else result + "ing"

    if orig_word and orig_word[0].isupper():
        result = result.capitalize()

    trail = re.search(r"[^\w]+$", orig_word)
    if trail:
        result = re.sub(r"[^\w]+$", "", result) + trail.group()

    return result


_ASCII_RE = re.compile(r"^[\x00-\x7F]+$")

def _is_ascii_english(token: str) -> bool:
    """Return True if the token contains only ASCII characters."""
    return bool(_ASCII_RE.match(token))


def _fluency_ok(
    orig_text:  str,
    trial_text: str,
    orig_word:  str,
    sub_word:   str,
    orig_emb:   np.ndarray,
    orig_ppl:   float,
    sim_threshold: float,
    ppl_factor:    float,
) -> Tuple[bool, float, float]:
    if not _is_ascii_english(sub_word):
        return False, 0.0, 0.0

    if not _pos_ok(orig_word, sub_word, orig_text, trial_text):
        return False, 0.0, 0.0

    if not _wordnet_ok(orig_word, sub_word):
        return False, 0.0, 0.0

    s = float(orig_emb @ _embed([trial_text])[0])
    if s < sim_threshold:
        return False, s, 0.0

    orig_local_ppl = _local_perplexity(orig_text, orig_word)
    ok, trial_ppl  = _ppl_ok(trial_text, sub_word, orig_ppl,
                              orig_local_ppl, ppl_factor)
    if not ok:
        return False, s, trial_ppl

    return True, s, trial_ppl



def _print_step(step: int, text: str, r: float, s: float,
               orig_ppl: float, trial_ppl: float, change: str):
    # Convert log-odds back to a probability just for display
    p_yes     = torch.sigmoid(torch.tensor(r)).item()
    direction = f"YES ({p_yes:.2%})" if r >= 0 else f"NO ({1-p_yes:.2%})"
    ppl_delta = trial_ppl - orig_ppl
    ppl_str   = f"+{ppl_delta:.1f}" if ppl_delta >= 0 else f"{ppl_delta:.1f}"
    print(f"\n── step {step}"
          f"  log-odds={r:+.3f} → {direction}"
          f"  sim={s:.4f}  ppl={trial_ppl:.1f} ({ppl_str})  ({change})")
    print(text)

def _find_text_token_span(text: str) -> Tuple[int, int, torch.Tensor]:
    """
    Find where `text` sits inside the full Qwen chat-formatted prompt,
    working in character space (robust to subword retokenisation artifacts).

    Returns (text_start, text_end, text_ids_within_prompt) where
    text_start/end are token indices into the full prompt.
    """
    # Build the full prompt string and tokenise it with char offsets
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": VOTER_PROFILE + "\n\n" + PROP_ANCHOR + text},
    ]
    prompt_str = _qwen_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + '{"vote": "'

    # Locate the text inside the prompt string
    char_start = prompt_str.find(text)
    if char_start == -1:
        char_start = prompt_str.find(text[:30].strip())
    if char_start == -1:
        raise ValueError(
            "Could not locate text inside the prompt string. "
            "Check that VOTER_PROFILE/SYSTEM_PROMPT don't contain the same substring."
        )
    char_end = char_start + len(text)

    enc = _qwen_tok(
        prompt_str, return_tensors="pt",
        return_offsets_mapping=True,
    ).to(DEVICE)

    offsets    = enc["offset_mapping"][0]   # (seq_len, 2)
    input_ids  = enc["input_ids"]

    tok_start, tok_end = None, None
    for i, (cs, ce) in enumerate(offsets.tolist()):
        if cs == 0 and ce == 0:
            continue   # special token
        if tok_start is None and cs >= char_start:
            tok_start = i
        if ce <= char_end:
            tok_end = i + 1

    if tok_start is None or tok_end is None:
        raise ValueError("Could not map text characters to token indices.")

    return tok_start, tok_end, input_ids


def whitebox_attack(
    start_text:    str,
    sim_threshold: float = 0.85,
    ppl_factor:    float = 2.0,
    max_iters:     int   = 20,
    top_k:         int   = 20,   # top-k vocab candidates per token position
    return_history: bool = False,
):
    _init_reward_model()
    _init_embedder()
    _init_gpt2()

    vocab_emb   = _qwen_model.model.embed_tokens.weight.detach()
    special_ids = set(_qwen_tok.all_special_ids)

    orig_emb    = _embed([start_text])[0]
    orig_ppl    = _perplexity(start_text)
    base_r, _, full_ids = reward(start_text, with_grad=False)
    _, _, full_ids = reward(start_text, with_grad=True)
    MARKER = PROP_ANCHOR

    # Flip to opposite of current vote
    flip_to_yes = base_r < 0   # NO→YES if True, YES→NO if False

    def _decode_with_swap(cur_full_ids: torch.Tensor, swap_pos: int,
                          swap_tok_id: int) -> Optional[str]:
        """Swap one token in the full prompt ids and extract the text portion."""
        trial_ids = cur_full_ids.clone()
        trial_ids[0, swap_pos] = swap_tok_id
        decoded = _qwen_tok.decode(trial_ids[0], skip_special_tokens=False)
        # Remove all special tokens text representations
        decoded = re.sub(r"<\|[^|]+\|>", "", decoded).strip()
        marker_pos = decoded.find(MARKER)
        if marker_pos == -1:
            return None
        text_portion = decoded[marker_pos + len(MARKER):]
        # Stop at any role boundary that might leak through
        for boundary in ["\nassistant", "\nuser", "\nsystem", "assistant", "<im_end>"]:
            bp = text_portion.find(boundary)
            if bp != -1:
                text_portion = text_portion[:bp]
        return text_portion.strip()

    print(f"\nOriginal  log-odds={base_r:+.3f} (YES if >0, NO if <0)  ppl={orig_ppl:.1f}")
    print(start_text)

    current_text = start_text
    current_r    = base_r
    swap_history = []   # list of {"step", "from", "to", "log_odds"}

    for step in range(max_iters):
        r, grad, cur_full_ids = reward(current_text, with_grad=True)

        cur_start, cur_end, _ = _find_text_token_span(current_text)
        text_ids = cur_full_ids[:, cur_start:cur_end]
        text_len = text_ids.shape[1]

        text_grad = grad[cur_start:cur_end]       # (text_len, hidden)
        scores    = text_grad @ vocab_emb.T        # (text_len, vocab)
        if not flip_to_yes:
            scores = -scores   # gradient points toward YES; negate to go toward NO

        candidates = []   # (abs_prompt_pos, tok_id, orig_tok_str, new_tok_str)
        for pos in range(text_len):
            if text_ids[0, pos].item() in special_ids:
                continue
            orig_tok_str = _qwen_tok.decode(
                [text_ids[0, pos].item()]
            ).strip().lstrip("\u2581").strip()
            if not re.search(r"(?<![a-zA-Z])" + re.escape(orig_tok_str) + r"(?![a-zA-Z])",
                             current_text, re.IGNORECASE):
                continue
            topk = scores[pos].topk(top_k)
            for tok_id in topk.indices.tolist():
                new_tok_str = _qwen_tok.decode([tok_id]).strip().lstrip("\u2581").strip()
                if not re.fullmatch(r"[a-zA-Z]+", new_tok_str):
                    continue
                if new_tok_str.lower() not in _ENGLISH_WORDS:
                    continue
                if new_tok_str.lower() == orig_tok_str.lower():
                    continue
                candidates.append((cur_start + pos, tok_id, orig_tok_str, new_tok_str))

        surviving = []
        for abs_pos, tok_id, orig_tok_str, new_tok_str in candidates:
            trial_text_raw = _decode_with_swap(cur_full_ids, abs_pos, tok_id)
            if trial_text_raw is None:
                continue
            if not re.search(r"(?<![a-zA-Z])" + re.escape(new_tok_str) + r"(?![a-zA-Z])",
                             trial_text_raw, re.IGNORECASE):
                continue
            inflected = _inflect(orig_tok_str, new_tok_str, current_text)
            escaped = re.escape(orig_tok_str)
            trial_text = re.sub(
                r"(?<![a-zA-Z])" + escaped + r"(?![a-zA-Z])",
                inflected, trial_text_raw, count=1, flags=re.IGNORECASE
            )

            ok, s, trial_ppl = _fluency_ok(current_text, trial_text, orig_tok_str,
                                            inflected, orig_emb, orig_ppl,
                                            sim_threshold, ppl_factor)
            if ok:
                surviving.append((abs_pos, tok_id, orig_tok_str, inflected,
                                   trial_text, s, trial_ppl))

        # ── Step 4: true reward eval for all survivors, pick global best ──────
        best_candidate = None
        best_trial_r   = current_r   # must strictly improve

        for abs_pos, tok_id, orig_tok_str, new_tok_str, trial_text, s, trial_ppl in surviving:
            trial_r = reward(trial_text)[0]
            improved = trial_r > best_trial_r if flip_to_yes else trial_r < best_trial_r
            if improved:
                best_trial_r   = trial_r
                best_candidate = (trial_text, orig_tok_str, new_tok_str, s, trial_ppl)

        if best_candidate is None:
            print(f"\n── step {step + 1}  no improving swap found, stopping.")
            break

        current_text, orig_tok_str, new_tok_str, s, trial_ppl = best_candidate
        current_r = best_trial_r
        _print_step(step + 1, current_text, current_r, s,
                    orig_ppl, trial_ppl, f"'{orig_tok_str}' \u2192 '{new_tok_str}'"
        )
        swap_history.append({
            "step": step + 1,
            "from": orig_tok_str,
            "to": new_tok_str,
            "log_odds": round(current_r, 4),
        })

        flipped_now = current_r > 0 if flip_to_yes else current_r < 0
        target_label = "YES" if flip_to_yes else "NO"
        if flipped_now:
            print(f"\n✓ Vote flipped to {target_label} at step {step + 1}!")
            break

    if return_history:
        return current_text, swap_history
    return current_text


def sentence_diffs(orig: str, final: str) -> list:
    """
    Return [{original, modified}] for every sentence that changed.
    Splits on sentence-ending punctuation within each line; blank lines
    (section headers, paragraph breaks) are kept as single units.
    """
    def split_sents(text):
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sent in re.split(r'(?<=[.!?])\s+', line):
                s = sent.strip()
                if s:
                    out.append(s)
        return out

    orig_s  = split_sents(orig)
    final_s = split_sents(final)
    diffs   = []
    matcher = difflib.SequenceMatcher(None, orig_s, final_s, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        diffs.append({
            "original": " ".join(orig_s[i1:i2]),
            "modified": " ".join(final_s[j1:j2]),
        })
    return diffs


def _split_paragraphs(text: str) -> list:
    """Split `text` into paragraphs (blocks separated by blank lines).
    Each `=== HEADER ===` line is treated as its own paragraph so section
    structure stays aligned between the original and modified texts."""
    paras, cur = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        if stripped.startswith("===") and stripped.endswith("==="):
            if cur:
                paras.append(" ".join(cur))
                cur = []
            paras.append(stripped)
            continue
        cur.append(stripped)
    if cur:
        paras.append(" ".join(cur))
    return paras


def _inline_word_diff(orig: str, mod: str) -> str:
    """Word-level diff of two strings rendered inline with change-tracking
    markup: unchanged words stay plain, removed words are ~~struck through~~
    and inserted words are **bolded**. This keeps every edit visible in its
    full paragraph context so semantic drift is easy to judge."""
    orig_w = orig.split()
    mod_w  = mod.split()
    sm  = difflib.SequenceMatcher(None, orig_w, mod_w, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(" ".join(orig_w[i1:i2]))
        elif tag == "replace":
            out.append("~~" + " ".join(orig_w[i1:i2]) + "~~ **"
                       + " ".join(mod_w[j1:j2]) + "**")
        elif tag == "delete":
            out.append("~~" + " ".join(orig_w[i1:i2]) + "~~")
        elif tag == "insert":
            out.append("**" + " ".join(mod_w[j1:j2]) + "**")
    return " ".join(out)


def paragraph_diffs(orig: str, final: str) -> list:
    """Return [{section, original, modified, inline_diff, word_changes}] for
    every paragraph that changed. Full paragraph context is preserved so it is
    clear whether a word swap distorts the surrounding meaning, and each edit is
    highlighted inline via `inline_diff`."""
    orig_p  = _split_paragraphs(orig)
    final_p = _split_paragraphs(final)

    # Nearest preceding `=== HEADER ===` for each original paragraph.
    sections, cur_section = [], ""
    for p in orig_p:
        if p.startswith("===") and p.endswith("==="):
            cur_section = p.strip("= ").strip()
        sections.append(cur_section)

    diffs   = []
    matcher = difflib.SequenceMatcher(None, orig_p, final_p, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        orig_block = "\n\n".join(orig_p[i1:i2])
        mod_block  = "\n\n".join(final_p[j1:j2])

        # Collect the individual word substitutions inside this paragraph.
        word_changes = []
        sm = difflib.SequenceMatcher(None, orig_block.split(),
                                     mod_block.split(), autojunk=False)
        for t, a1, a2, b1, b2 in sm.get_opcodes():
            if t == "equal":
                continue
            word_changes.append({
                "from": " ".join(orig_block.split()[a1:a2]),
                "to":   " ".join(mod_block.split()[b1:b2]),
            })

        diffs.append({
            "section":      sections[i1] if i1 < len(sections) else "",
            "original":     orig_block,
            "modified":     mod_block,
            "inline_diff":  _inline_word_diff(orig_block, mod_block),
            "word_changes": word_changes,
        })
    return diffs


def write_markdown_report(proposition: str, results: list, path: str,
                          settings: dict):
    """Write a human-readable markdown report with paragraph-level, inline
    change-tracked diffs for each voter — easier to eyeball than the JSON."""
    lines = []
    lines.append("# RL Attack — Paragraph-Level Diffs vs Original Proposition")
    lines.append("")
    lines.append(f"**Model:** Qwen2.5-7B-Instruct  ")
    lines.append(f"**Attack:** whitebox token-swap  ")
    lines.append(f"**Settings:** sim_threshold={settings['sim_threshold']}, "
                 f"ppl_factor={settings['ppl_factor']}, "
                 f"max_iters={settings['max_iters']}  ")
    lines.append(f"**Voters:** {len(results)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Voter | Base log-odds | Final log-odds | Flipped? | Swaps |")
    lines.append("|-------|:---:|:---:|:---:|:---:|")
    for r in results:
        flip = "✓" if r["flipped"] else "✗"
        lines.append(f"| {r['voter_row']} | {r['base_log_odds']:+.3f} | "
                     f"{r['final_log_odds']:+.3f} | {flip} | {r['n_swaps']} |")
    n_flipped = sum(r["flipped"] for r in results)
    lines.append("")
    lines.append(f"**Flip rate: {n_flipped}/{len(results)} "
                 f"({n_flipped / max(len(results),1):.0%})**")
    lines.append("")
    lines.append("---")

    for r in results:
        lines.append("")
        lines.append(f"## Voter {r['voter_row']} — "
                     f"{'flipped' if r['flipped'] else 'NOT flipped'}")
        lines.append(f"**Base log-odds:** {r['base_log_odds']:+.3f} → "
                     f"**Final:** {r['final_log_odds']:+.3f}  "
                     f"({r['n_swaps']} swaps)")
        lines.append("")

        if r["swaps"]:
            lines.append("| Step | From | To | After log-odds |")
            lines.append("|------|------|----|---:|")
            for s in r["swaps"]:
                lines.append(f"| {s['step']} | `{s['from']}` | `{s['to']}` | "
                             f"{s['log_odds']:+.3f} |")
            lines.append("")

        pdiffs = r.get("paragraph_diffs", [])
        if not pdiffs:
            lines.append("*No paragraph-level changes.*")
            continue

        lines.append("**Changed paragraphs** "
                     "(~~removed~~ / **inserted**):")
        lines.append("")
        for d in pdiffs:
            if d["section"]:
                lines.append(f"### {d['section']}")
            lines.append("")
            lines.append("> " + d["inline_diff"].replace("\n\n", "\n>\n> "))
            lines.append("")
        lines.append("---")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial text attacks")
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--sim_threshold", type=float, default=0.85,
                   help="Min cosine similarity to original (default: 0.85)")
    p.add_argument("--ppl_factor", type=float, default=2.0,
                   help="Max allowed perplexity increase factor (default: 2.0)")
    p.add_argument("--max_iters", type=int, default=20)
    p.add_argument("--top_k", type=int, default=50)
    return p.parse_args()


# def main():
#     args = parse_args()

#     text = args.text or (
#         "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say "
#         "that they were perfectly normal, thank you very much. Mr. Dursley made "
#         "drills. He was a big, beefy man with hardly any neck, although he did "
#         "have a very large moustache. Mrs. Dursley was thin and blonde and had "
#         "twice the usual amount of neck, which came in very useful as she spent "
#         "so much of her time spying on the neighbours. The Dursleys had a small "
#         "son called Dudley and in their opinion there was no finer boy anywhere."
#     )

#     print(f"\n{'═'*70}")
#     print(f"Whitebox Attack  |  sim_threshold={args.sim_threshold}"
#           f"  |  ppl_factor={args.ppl_factor}  |  max_iters={args.max_iters}")
#     print(f"{'═'*70}")

#     final = whitebox_attack(
#         text,
#         sim_threshold=args.sim_threshold,
#         ppl_factor=args.ppl_factor,
#         max_iters=args.max_iters,
#     )

#     print(f"\n{'═'*70}")
#     print("FINAL TEXT")
#     print(f"{'═'*70}")
#     print(final)
def main():
    args = parse_args()
    df = pd.read_csv("anes2024.csv", low_memory=False).reset_index(drop=True)

    with open("prop_minwage_full.txt") as f:
        proposition = f.read().strip()

    results = []
    results_path = "rl_results.json"

    global VOTER_PROFILE
    for row_idx, row in df.iterrows():
        if row_idx >= 10:   # remove this to run all voters
            break
        VOTER_PROFILE = build_voter_profile(row, ALL_QUESTIONS)

        print(f"\n{'═'*70}")
        print(f"Voter {row_idx}  |  sim_threshold={args.sim_threshold}"
              f"  |  ppl_factor={args.ppl_factor}  |  max_iters={args.max_iters}")
        print(f"{'═'*70}")

        base_r = reward(proposition)[0]

        final, swap_history = whitebox_attack(
            proposition,
            sim_threshold=args.sim_threshold,
            ppl_factor=args.ppl_factor,
            max_iters=args.max_iters,
            return_history=True,
        )

        final_r = reward(final)[0]
        flipped = (base_r < 0 and final_r > 0) or (base_r > 0 and final_r < 0)

        print(f"\n{'═'*70}")
        print(f"FINAL TEXT — Voter {row_idx}")
        print(f"{'═'*70}")
        print(final)

        results.append({
            "voter_row":       int(row_idx),
            "base_log_odds":   round(base_r, 4),
            "final_log_odds":  round(final_r, 4),
            "flipped":         flipped,
            "n_swaps":         len(swap_history),
            "swaps":           swap_history,   # [{"step","from","to","log_odds"}, ...]
            "sentence_diffs":  sentence_diffs(proposition, final),
            "paragraph_diffs": paragraph_diffs(proposition, final),
            "final_text":      final,
        })

        with open(results_path, "w") as f:
            json.dump({"proposition": proposition, "results": results}, f, indent=2)

    settings = {
        "sim_threshold": args.sim_threshold,
        "ppl_factor":    args.ppl_factor,
        "max_iters":     args.max_iters,
    }
    md_path = "rl_results.md"
    write_markdown_report(proposition, results, md_path, settings)

    n_flipped = sum(r["flipped"] for r in results)
    print(f"\n=== DONE ===")
    print(f"  Voters:  {len(results)}")
    print(f"  Flipped: {n_flipped}/{len(results)}")
    print(f"  Saved word- + paragraph-level diffs + final texts → {results_path}")
    print(f"  Saved readable paragraph diff report → {md_path}")


if __name__ == "__main__":
    main()