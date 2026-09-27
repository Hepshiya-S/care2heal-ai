from agents.rag_pipeline import llm
import json

# Layer 1: fast, deterministic keyword matching.
# These are common ways people describe genuine emergencies —
# not exhaustive, but a strong, instant first pass.
RED_FLAG_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "severe bleeding", "won't stop bleeding",
    "unconscious", "passed out", "fainted", "seizure", "stroke",
    "face drooping", "slurred speech", "severe allergic reaction",
    "swelling of face", "throat closing", "can't wake up", "overdose",
    "fell and can't get up", "severe confusion", "sudden confusion",
    "blue lips", "chest tightness"
]


def keyword_red_flag_check(text: str) -> bool:
    """Layer 1: instant, guaranteed catch for known emergency phrases."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in RED_FLAG_KEYWORDS)


def llm_red_flag_check(text: str) -> bool:
    """Layer 2: catches paraphrased or unusual descriptions the keyword list would miss."""
    prompt = f"""You are a safety classifier for a medical assistant app used by elderly patients.

Does the message below describe a potential medical EMERGENCY (e.g. chest pain, severe
allergic reaction, stroke symptoms, severe injury, loss of consciousness, severe bleeding,
overdose)? Reply with only one word: "yes" or "no".

Message: {text}
"""
    response = llm.invoke(prompt)
    return response.content.strip().lower().startswith("yes")


def check_emergency(text: str) -> bool:
    """
    Combined check — flags as emergency if EITHER layer says yes.
    This is deliberately biased toward false positives over false negatives:
    for safety, wrongly flagging a non-emergency is far preferable to
    missing a real one.
    """
    if keyword_red_flag_check(text):
        return True
    return llm_red_flag_check(text)


def load_drug_data() -> list:
    with open("data/drug_data.json", "r") as f:
        return json.load(f)


def check_drug_interactions(drug_names: list) -> list:
    """
    Given a list of drug names a person is taking, checks each drug's known
    interactions against every OTHER drug in the list.
    Returns a list of plain-language warning strings.
    """
    drug_data = load_drug_data()
    warnings = []

    # Build a quick lookup: drug name (lowercase) -> full entry
    drug_lookup = {entry["name"].lower(): entry for entry in drug_data}

    lowered_names = [name.lower() for name in drug_names]

    for i, drug_name in enumerate(lowered_names):
        entry = drug_lookup.get(drug_name)
        if not entry:
            continue  # drug not in our knowledge base, skip silently

        for interaction_text in entry["interactions"]:
            # Check if any OTHER drug the person takes is mentioned in this interaction
            for j, other_name in enumerate(lowered_names):
                if i == j:
                    continue
                if other_name in interaction_text.lower():
                    warnings.append(
                        f"⚠ {entry['name']} may interact with {drug_names[j]}: {interaction_text}"
                    )

    # Remove duplicate warnings (A-vs-B and B-vs-A both trigger the same pair)
    return list(dict.fromkeys(warnings))


if __name__ == "__main__":
    # Quick manual test using two drugs we know interact from Day 2's data
    test_meds = ["Warfarin", "Aspirin", "Paracetamol"]
    print(f"Checking interactions for: {test_meds}\n")
    results = check_drug_interactions(test_meds)
    if results:
        for warning in results:
            print(warning)
    else:
        print("No known interactions found.")