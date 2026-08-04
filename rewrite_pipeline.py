"""Adversarial-rewrite evaluation pipeline.

Pipeline:
    original -> attack -> adversarial rewrite
    (original, adversarial rewrite) -> semantic-fidelity log-odds
    adversarial rewrite -> paraphrases -> vote-agreement robustness

By default the attack block calls the section-restricted ``whitebox_attack``
from ``rl.py``. Pass a completed rewrite with ``--adv_rewrite`` to skip the
attack and run only the similarity and robustness blocks.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import rl


def section_parts(text: str, header: str) -> tuple[str, str, str]:
    """Return text before, inside, and after a === SECTION === block."""
    marker = f"=== {header} ==="
    start = text.index(marker) + len(marker)
    next_header = text.find("\n===", start)
    end = len(text) if next_header == -1 else next_header
    return text[:start], text[start:end], text[end:]


def attack(original: str, direction: str = "auto") -> str:
    """Run the section-restricted white-box attack from ``rl.py``."""
    return rl.whitebox_attack(
        original,
        direction=direction,
    )


def vote_score(text: str) -> float:
    """Return voter YES-vs-NO log-odds for ``text``."""
    return rl.reward(text, with_grad=False)[0]


def vote_from_score(score: float) -> str:
    """Convert YES-vs-NO log-odds into the corresponding vote."""
    return "YES" if score >= 0 else "NO"


def robustness_check(
    adv_rewrite: str,
    n_paraphrases: int = 100,
    temperature: float = 1.0,
    section_header: Optional[str] = None,
    vote_fn: Callable[[str], float] = vote_score,
) -> Dict:
    reference_score = vote_fn(adv_rewrite)
    reference_vote = vote_from_score(reference_score)
    trials = []

    if section_header:
        before, paraphrase_target, after = section_parts(
            adv_rewrite,
            section_header,
        )
    else:
        before, paraphrase_target, after = "", adv_rewrite, ""

    for index in range(n_paraphrases):
        rewritten_target = rl.paraphrase(
            paraphrase_target.strip(),
            temperature=temperature,
        )
        rewritten = (
            before + "\n" + rewritten_target + "\n" + after.lstrip("\n")
            if section_header
            else rewritten_target
        )
        score = vote_fn(rewritten)
        vote = vote_from_score(score)
        trials.append({
            "index": index + 1,
            "text": rewritten,
            "vote_score": score,
            "vote": vote,
            "agrees": vote == reference_vote,
        })

    agreement_count = sum(trial["agrees"] for trial in trials)
    return {
        "reference_vote_score": reference_score,
        "reference_vote": reference_vote,
        "paraphrased_section": section_header,
        "n_paraphrases": n_paraphrases,
        "agreement_count": agreement_count,
        "agreement_rate": (
            agreement_count / n_paraphrases if n_paraphrases else 1.0
        ),
        "trials": trials,
    }


def evaluate_rewrite(
    original: str,
    adv_rewrite: Optional[str] = None,
    n_paraphrases: int = 100,
    temperature: float = 1.0,
    direction: str = "auto",
    section_header: Optional[str] = None,
    attack_fn: Callable[[str, str], str] = attack,
) -> Dict:
    """Run the attack, semantic-fidelity, and robustness blocks."""
    direction = direction.lower()
    if direction not in {"yes", "no", "auto"}:
        raise ValueError("direction must be 'yes', 'no', or 'auto'")

    original_vote_score = vote_score(original)
    effective_direction = (
        ("yes" if original_vote_score < 0 else "no")
        if direction == "auto"
        else direction
    )
    if section_header is None:
        section_header = (
            "SUPPORTERS" if effective_direction == "yes" else "OPPONENTS"
        )

    if adv_rewrite is None:
        adv_rewrite = attack_fn(original, direction)

    similarity_score = rl.similarity_check(original, adv_rewrite)
    robustness = robustness_check(
        adv_rewrite,
        n_paraphrases=n_paraphrases,
        temperature=temperature,
        section_header=section_header,
    )
    adv_rewrite_vote_score = robustness["reference_vote_score"]
    original_vote = vote_from_score(original_vote_score)
    adv_rewrite_vote = vote_from_score(adv_rewrite_vote_score)

    return {
        "original": original,
        "adv_rewrite": adv_rewrite,
        "requested_direction": direction,
        "effective_direction": effective_direction,
        "editable_section": section_header,
        "similarity_score": similarity_score,
        "original_vote": {
            "vote": original_vote,
            "yes_no_log_odds": original_vote_score,
        },
        "adv_rewrite_vote": {
            "vote": adv_rewrite_vote,
            "yes_no_log_odds": adv_rewrite_vote_score,
            "log_odds_delta_from_original": (
                adv_rewrite_vote_score - original_vote_score
            ),
            "flipped_from_original": adv_rewrite_vote != original_vote,
        },
        "robustness": robustness,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an adversarial ballot rewrite."
    )
    parser.add_argument(
        "--original",
        default="prop_minwage_full.txt",
        help="Path to the original ballot text.",
    )
    parser.add_argument(
        "--adv_rewrite",
        default=None,
        help="Path to a precomputed adversarial rewrite. If omitted, attack() runs.",
    )
    parser.add_argument(
        "--direction",
        choices=["yes", "no", "auto"],
        default="auto",
        help=(
            "Attack target: yes edits SUPPORTERS, no edits OPPONENTS, "
            "auto targets the opposite of the original vote."
        ),
    )
    parser.add_argument(
        "--voter_profile",
        default=None,
        help="Optional path to a prebuilt voter-profile text file.",
    )
    parser.add_argument("--n_paraphrases", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default="rewrite_evaluation.json")
    return parser.parse_args()


def main():
    args = parse_args()
    original = Path(args.original).read_text().strip()
    adv_rewrite = (
        Path(args.adv_rewrite).read_text().strip()
        if args.adv_rewrite
        else None
    )

    if args.voter_profile:
        rl.VOTER_PROFILE = Path(args.voter_profile).read_text().strip()

    result = evaluate_rewrite(
        original=original,
        adv_rewrite=adv_rewrite,
        n_paraphrases=args.n_paraphrases,
        temperature=args.temperature,
        direction=args.direction,
    )
    Path(args.output).write_text(json.dumps(result, indent=2))

    robustness = result["robustness"]
    original_vote = result["original_vote"]
    rewrite_vote = result["adv_rewrite_vote"]
    print(f"Similarity YES-vs-NO log-odds: {result['similarity_score']:+.4f}")
    print(
        f"Original vote: {original_vote['vote']} "
        f"({original_vote['yes_no_log_odds']:+.4f})"
    )
    print(
        f"Adversarial rewrite vote: {rewrite_vote['vote']} "
        f"({rewrite_vote['yes_no_log_odds']:+.4f}); "
        f"delta={rewrite_vote['log_odds_delta_from_original']:+.4f}; "
        f"flipped={rewrite_vote['flipped_from_original']}"
    )
    print(
        f"Paraphrase vote agreement: {robustness['agreement_count']}/"
        f"{robustness['n_paraphrases']} "
        f"({robustness['agreement_rate']:.1%})"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
