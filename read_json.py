import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from collections import OrderedDict

# ---------- CONFIG ----------
INPUT_FOLDER = Path("json_inputs")   # Folder with multiple JSON files
OUTPUT_FOLDER = Path("parsed_outputs")


def extract_text_and_math(html_content):
    """Extract plain text and math expressions from HTML."""
    if not html_content:
        return "", []
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # Extract inline ($...$) and display ($$...$$) LaTeX math
    math_expressions = re.findall(r"\${1,2}(.*?)\${1,2}", html_content, re.DOTALL)
    math_expressions = [m.replace("$", "").strip() for m in math_expressions]

    # Remove duplicates but keep order
    math_expressions = list(OrderedDict.fromkeys(math_expressions))
    return text, math_expressions


def extract_svgs(html_content):
    """Extract all SVG blocks if present."""
    if not html_content:
        return []
    svgs = re.findall(r"(<svg.*?</svg>)", html_content, re.DOTALL)
    return svgs


def normalize_record(record):
    """Normalize a single exercise record."""
    grade = record.get("grade", "").strip()
    lesson_number = str(record.get("lesson_number", "")).strip()
    topic = record.get("topic", "").strip()
    exercise_type = record.get("exercise_type", "").strip()
    exercise_number = str(record.get("exercise_number", "")).strip()

    canonical_id = f"{grade}_{lesson_number}_{exercise_type}_{exercise_number}".replace(" ", "_")

    all_texts = {}
    all_math = []

    for section in record.get("sections", []):
        for field in ["question_html", "solution_html", "hint_html", "full_solution_html"]:
            html = section.get(field, "")
            text, math_exprs = extract_text_and_math(html)
            if text:
                all_texts.setdefault(field.replace("_html", ""), []).append(text)
            all_math.extend(math_exprs)

    # Extract SVGs from any HTML field (data_html or combined sections)
    svg_content = extract_svgs(record.get("data_html", ""))
    
    # Remove duplicates in SVGs while preserving order
    svg_content = list(OrderedDict.fromkeys(svg_content))

    return {
        "canonical_exercise_id": canonical_id,
        "grade": grade,
        "lesson_number": lesson_number,
        "topic": topic,
        "exercise_type": exercise_type,
        "exercise_number": exercise_number,
        "text": all_texts,
        "math": all_math,
        "svg": svg_content,
        "metadata": {
            "id": record.get("id"),
            "created_date": record.get("created_date"),
            "updated_date": record.get("updated_date"),
            "created_by": record.get("created_by"),
            "is_sample": record.get("is_sample", False)
        }
    }


def process_json_file(file_path, output_folder):
    """Process a single JSON file into parsed format."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Save raw copy
    raw_out_path = output_folder / f"{file_path.stem}_raw.json"
    with open(raw_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Parse and save normalized version
    parsed_records = [normalize_record(rec) for rec in data]
    parsed_out_path = output_folder / f"{file_path.stem}_parsed.json"
    with open(parsed_out_path, "w", encoding="utf-8") as f:
        json.dump(parsed_records, f, ensure_ascii=False, indent=2)

    print(f"✅ Processed {file_path.name} → {parsed_out_path.name}")
    return parsed_records


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    files = list(INPUT_FOLDER.glob("*.json"))
    if not files:
        print("⚠️ No JSON files found in", INPUT_FOLDER.resolve())
        return

    all_parsed = []
    for json_file in files:
        parsed_records = process_json_file(json_file, OUTPUT_FOLDER)
        all_parsed.extend(parsed_records)

    # Save combined parsed data
    combined_path = OUTPUT_FOLDER / "all_parsed.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_parsed, f, ensure_ascii=False, indent=2)

    print(f"\n📦 Combined all parsed records into {combined_path.name}")


if __name__ == "__main__":
    main()