from agents.rag_pipeline import llm


def generate_daily_checklist(medicines: list) -> str:
    """
    Turns a list of medicines with instructions into a simple daily checklist,
    grouped by time of day, in plain language for an elderly user.
    """
    if not medicines:
        return "No medicines on file yet."

    meds_text = "\n".join(
        f"- {m['name']} {m['dosage']}: {m['instructions']}" for m in medicines
    )

    prompt = f"""Turn the medicine list below into a simple daily checklist grouped by
time of day (Morning, Afternoon, Evening, Night). Use plain, friendly language suitable
for an elderly person. If a medicine's timing isn't clear, make a reasonable guess based
on common practice (e.g. "twice daily" usually means morning and evening).

Medicines:
{meds_text}

Format as a short list under each time-of-day heading."""
    response = llm.invoke(prompt)
    return response.content