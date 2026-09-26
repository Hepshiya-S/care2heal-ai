
import pytesseract
from PIL import Image, ImageOps
import json
from agents.rag_pipeline import llm  # reuse the same LLM connection from Day 3

# Windows only: if 'tesseract --version' failed earlier, uncomment and
# point this to wherever your installer put it.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Basic preprocessing to help Tesseract: convert to grayscale,
    increase contrast, and upscale — all proven to improve OCR accuracy
    on real-world document photos.
    """
    image = image.convert("L")  # grayscale
    image = ImageOps.autocontrast(image)
    # Upscale 2x — Tesseract performs better on higher-resolution text
    image = image.resize((image.width * 2, image.height * 2))
    return image


def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    image = preprocess_image(image)
    raw_text = pytesseract.image_to_string(image)
    return raw_text.strip()


def parse_prescription_text(raw_text: str) -> list:
    """
    Takes messy OCR text, asks the LLM to extract structured medicine entries.
    Returns a list of dicts: [{"name": ..., "dosage": ..., "instructions": ...}, ...]
    """
    prompt = f"""Extract every medicine mentioned in the text below into a JSON array.
Each item must have exactly these fields: "name", "dosage", "instructions".

The text comes from OCR and may contain misspelled drug names due to scanning errors
(e.g. "Poracetamol" is likely "Paracetamol", "Noproxen" is likely "Naproxen").
If a name looks like a probable OCR misspelling of a well-known medication, correct it
to the standard spelling. If a field isn't mentioned, use an empty string.
Return ONLY valid JSON, nothing else — no explanation, no markdown code fences.

Text:
{raw_text}
"""
    response = llm.invoke(prompt)
    raw_output = response.content.strip()

    # Safety: strip markdown fences if the model adds them anyway
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print("Warning: could not parse LLM output as JSON. Raw output was:")
        print(raw_output)
        return []


if __name__ == "__main__":
    test_image_path = input("Enter path to a test image (e.g. data/sample_prescription.jpg): ")

    raw_text = extract_text_from_image(test_image_path)
    print("\n--- Raw OCR text ---\n")
    print(raw_text)

    medicines = parse_prescription_text(raw_text)
    print("\n--- Parsed medicines ---\n")
    for med in medicines:
        print(med)




