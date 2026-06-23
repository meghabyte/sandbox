"""
voter_profiles.py
=================
Shared utilities for building voter profiles from ANES 2024 survey data.
Imported by all simulation and analysis scripts.

ANES 2024 data: https://electionstudies.org/data-center/2024-time-series-study/
"""

import json
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# ANES survey questions used to build voter profiles
# Each entry: (anes_column, human_readable_label, value_map or None)
# value_map: dict mapping coded integers to readable strings
# None = use raw numeric value (e.g. thermometer scales)
# ---------------------------------------------------------------------------

IDENTITY_QUESTIONS = [
    ("V241003",  "Gender",
     {1: "Man", 2: "Woman"}),
    ("V241501x", "Race/ethnicity",
     {1: "White, non-Hispanic", 2: "Black, non-Hispanic", 3: "Hispanic",
      4: "Asian or Native Hawaiian/other Pacific Islander, non-Hispanic",
      5: "Native American/Alaska Native or other race, non-Hispanic",
      6: "Multiple races, non-Hispanic"}),
    ("V241463",  "Highest level of education",
     {1: "Less than 1st grade", 2: "1st–4th grade", 3: "5th–6th grade",
      4: "7th–8th grade", 5: "9th grade", 6: "10th grade", 7: "11th grade",
      8: "12th grade, no diploma",
      9: "High school graduate or equivalent (e.g. GED)",
      10: "Some college, no degree",
      11: "Associate degree – vocational/occupational",
      12: "Associate degree – academic",
      13: "Bachelor's degree (BA, BS, etc.)",
      14: "Master's degree (MA, MS, MBA, etc.)",
      15: "Professional degree (MD, JD, etc.)",
      16: "Doctorate (PhD, EdD, etc.)",
      95: "Other"}),
    ("V241567x", "Household income",
     {1: "Under $9,999", 2: "$10,000–$29,999", 3: "$30,000–$59,999",
      4: "$60,000–$99,999", 5: "$100,000–$249,999", 6: "$250,000 or more"}),
    ("V241422",  "Religion",
     {1: "Protestant", 2: "Roman Catholic", 3: "Orthodox Christian",
      4: "Latter-Day Saints (LDS)", 5: "Jewish", 6: "Muslim",
      7: "Buddhist", 8: "Hindu", 9: "Atheist", 10: "Agnostic",
      11: "Something else", 12: "Nothing in particular"}),
    ("V241025",  "Party registration",
     {1: "Democratic Party", 2: "Republican Party",
      4: "None or independent", 5: "Another party"}),
]

ISSUE_QUESTIONS = [
    # --- Political ideology ---
    ("V241177",  "Liberal-conservative self-placement (1=extremely liberal, 7=extremely conservative)", None),
    ("V241226",  "Party identification (1=strong Democrat, 7=strong Republican)", None),

    # --- Economy ---
    ("V241451",  "Personal finances compared to one year ago",
     {1: "Much better off", 2: "Somewhat better off", 3: "About the same",
      4: "Somewhat worse off", 5: "Much worse off"}),
    ("V241291",  "National economy compared to one year ago",
     {1: "Gotten better", 2: "Stayed about the same", 3: "Gotten worse"}),


    # --- Environment & climate ---
    ("V241282",  "Federal spending on protecting the environment",
     {1: "Increased", 2: "Decreased", 3: "Kept the same"}),

    # --- Government role & spending ---
    ("V241261",  "Federal spending on Social Security",
     {1: "Increased", 2: "Decreased", 3: "Kept the same"}),
    ("V242351",  "Government spending to help people pay for health insurance",
     {1: "Increase", 2: "Decrease", 3: "No change"}),
    ("V241273",  "Federal spending on welfare programs",
     {1: "Increased", 2: "Decreased", 3: "Kept the same"}),

    # --- Abortion ---
    ("V241302",  "Abortion policy view",
     {1: "Should never be permitted by law",
      2: "Permitted only in cases of rape, incest, or danger to woman's life",
      3: "Permitted beyond rape/incest but only after need clearly established",
      4: "Woman should always be able to obtain abortion as personal choice",
      5: "Other"}),
    ("V242176",  "Importance of abortion as an issue",
     {1: "Extremely important", 2: "Very important", 3: "Moderately important",
      4: "Slightly important", 5: "Not at all important"}),

    # --- Guns ---
    ("V242175",  "Importance of gun policy as an issue",
     {1: "Extremely important", 2: "Very important", 3: "Moderately important",
      4: "Slightly important", 5: "Not at all important"}),
    ("V242325",  "Federal government should make it easier or harder to buy a gun",
     {1: "More difficult", 2: "Easier", 3: "Keep rules about the same"}),
    ("V242326",  "Background checks for gun purchases at gun shows / private sales",
     {1: "Favor", 2: "Oppose", 3: "Neither favor nor oppose"}),
    ("V241242",  "Defense spending self-placement (1=greatly decrease, 7=greatly increase)", None),

    # --- Foreign policy ---
    ("V241404",  "Favor or oppose U.S. providing humanitarian aid to Palestinians in Gaza",
     {1: "Favor", 2: "Oppose", 3: "Neither favor nor oppose"}),
    ("V242178",  "Importance of war in Gaza as an issue",
     {1: "Extremely important", 2: "Very important", 3: "Moderately important",
      4: "Slightly important", 5: "Not at all important"}),

    # --- Race ---
    ("V242181",  "Importance of racial inequality as an issue",
     {1: "Extremely important", 2: "Very important", 3: "Moderately important",
      4: "Slightly important", 5: "Not at all important"}),
    ("V242242",  "Affirmative action in university admissions",
     {1: "Favor", 2: "Oppose", 3: "Neither favor nor oppose"}),

    # --- Immigration ---
    ("V241267",  "Federal spending on tightening border security",
     {1: "Increased", 2: "Decreased", 3: "Kept the same"}),
    ("V241386",  "Policy toward unauthorized immigrants",
     {1: "Make felons and send back to home country",
      2: "Guest worker program — stay to work for limited time",
      3: "Allow to remain and qualify for citizenship if requirements met",
      4: "Allow to remain and qualify for citizenship without penalties"}),
    ("V241387",  "End birthright citizenship for children of unauthorized immigrants",
     {1: "Favor", 2: "Oppose", 3: "Neither favor nor oppose"}),
    # --- Healthcare & vaccines ---
    ("V242317",  "Require children to be vaccinated to attend public schools",
     {1: "Favor", 2: "Oppose", 3: "Neither favor nor oppose"}),
    ("V242354",  "Do health benefits of vaccinations outweigh the risks",
     {1: "Benefits outweigh risks", 2: "Risks outweigh benefits", 3: "No difference"}),
]

AFFECT_QUESTIONS = [
    # --- Racial/ethnic group thermometers (0-100) ---
    ("V242517", "Feeling thermometer: Illegal immigrants (0=very cold, 100=very warm)", None),
    ("V242518", "Feeling thermometer: Whites (0=very cold, 100=very warm)", None),
    ("V242516", "Feeling thermometer: Blacks (0=very cold, 100=very warm)", None),
    ("V242515", "Feeling thermometer: Hispanics (0=very cold, 100=very warm)", None),
    ("V242514", "Feeling thermometer: Asian-Americans (0=very cold, 100=very warm)", None),

    # --- Other group thermometers (0-100) ---
    ("V242155", "Feeling thermometer: Rural Americans (0=very cold, 100=very warm)", None),
    ("V242150", "Feeling thermometer: Police (0=very cold, 100=very warm)", None),
    ("V242151", "Feeling thermometer: Transgender people (0=very cold, 100=very warm)", None),
    ("V242146", "Feeling thermometer: Muslims (0=very cold, 100=very warm)", None),
    ("V242144", "Feeling thermometer: Gay men and lesbians (0=very cold, 100=very warm)", None),
    ("V242138", "Feeling thermometer: Feminists (0=very cold, 100=very warm)", None),
    ("V242149", "Feeling thermometer: Jews (0=very cold, 100=very warm)", None),

    # --- Identity ---
    ("V242512", "Self-identification as feminist or anti-feminist",
     {1: "Feminist", 2: "Anti-feminist", 3: "Neither"}),
]

ALL_QUESTIONS = IDENTITY_QUESTIONS + ISSUE_QUESTIONS + AFFECT_QUESTIONS

# Ground truth: 2024 presidential vote
VOTE_COLUMN = "V241039"
VOTE_MAP = {
    1: "Kamala Harris (Democrat)",
    2: "Donald Trump (Republican)",
    3: "Robert F. Kennedy Jr. (Independent)",
    4: "Cornel West (Independent)",
    5: "Jill Stein (Green)",
    6: "Another candidate",
}

# ANES missing value codes — treat as no answer
MISSING_CODES = {-9, -8, -7, -6, -5, -4, -3, -2, -1, 95, 98, 99, 998, 999}

# System prompts
PROP_SYSTEM_PROMPT = """You are simulating how a specific American voter would vote on a ballot proposition.
You will be given the voter's detailed survey responses, then the full text of a ballot proposition.
Based solely on this voter's profile, predict whether they would vote YES or NO on the proposition.

Respond with ONLY a JSON object in this exact format:
{"vote": "<YES or NO>"}

Where vote is exactly "YES" or "NO".
"""

PRESIDENTIAL_SYSTEM_PROMPT = """You are simulating the voting behavior of a specific American voter
based on their survey responses. You will be given detailed information about this
voter's political views, values, demographics, and attitudes. Your task is to predict
how this voter would vote in the presidential election, staying true to their profile.

Respond with ONLY a JSON object in this exact format:
{"vote": "<candidate_party>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

Where vote is one of: "Democrat", "Republican", "Third party", "Would not vote"
"""


def build_voter_profile(row: pd.Series, questions: list) -> str:
    """Convert a single ANES respondent row into a readable text profile."""
    lines = ["=== VOTER PROFILE ===\n"]
    sections = [
        ("DEMOGRAPHICS AND IDENTITY", IDENTITY_QUESTIONS),
        ("POLICY POSITIONS", ISSUE_QUESTIONS),
        ("ATTITUDES AND FEELINGS (0=very cold/negative, 100=very warm/positive)", AFFECT_QUESTIONS),
    ]
    for section_name, section_questions in sections:
        lines.append(f"\n--- {section_name} ---")
        for col, label, value_map in section_questions:
            if col not in row.index:
                continue
            val = row[col]
            try:
                if pd.isna(val) or int(val) in MISSING_CODES:
                    continue
            except (ValueError, TypeError):
                continue
            if value_map is not None:
                readable = value_map.get(int(val), f"Code {int(val)}")
            else:
                readable = f"{val:.0f}" if isinstance(val, float) else str(int(val))
            lines.append(f"  {label}: {readable}")
    return "\n".join(lines)


def load_model(model_name: str):
    """
    Load a HuggingFace model with appropriate settings:
    - openai/gpt-oss-*  : native MXFP4 quantization (no extra config needed)
    - *70b / *72b       : 4-bit quantization via bitsandbytes (fits on 80GB GPU)
    - everything else   : float16
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if "gpt-oss" in model_name:
        print("Detected gpt-oss — using native MXFP4 quantization")
        # Leave headroom on each GPU for MXFP4→bf16 dequantization which
        # temporarily needs 2× shard size. 60GiB limit forces early distribution.
        n_gpus = torch.cuda.device_count()
        max_memory = {i: "60GiB" for i in range(n_gpus)}
        print(f"Spreading across {n_gpus} GPUs, max 60GiB each")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
            max_memory=max_memory,
        )
    elif any(x in model_name.lower() for x in ["70b", "72b"]):
        print("70B model detected — using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="cuda", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True,
        )

    return tokenizer, model
