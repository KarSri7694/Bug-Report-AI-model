import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List


CREATED_LINE_RE = re.compile(
    r"^\s*Created by\b.+?\bon\b\s+(?P<created>.+?)\s*$",
    re.IGNORECASE,
)

CREATED_SPLIT_RE = re.compile(r"^(?P<indent>\s*)Created by\b")

TEMPORAL_RE = re.compile(
    (
        r"(?P<weekday>[A-Za-z]+)\s+"
        r"(?P<month>[A-Za-z]+)\s+"
        r"(?P<day>\d{1,2})\s+"
        r"(?P<year>\d{4})\s+"
        r"(?P<hour>\d{1,2})\s+"
        r"(?P<minute>\d{1,2})\s+"
        r"(?P<second>\d{1,2})\s+"
        r"(?P<ampm>AM|PM)\s*"
        r"(?P<tz>[A-Za-z]{2,5})?"
    ),
    re.IGNORECASE,
)


PRIORITY_LABELS = ["critical", "high", "medium", "low", "unknown"]
SEVERITY_LABELS = ["critical", "major", "minor", "trivial", "unknown"]
TYPE_LABELS = [
    "bug",
    "regression",
    "feature",
    "build",
    "ui",
    "performance",
    "security",
    "unknown",
]
COMPONENT_LABELS = [
    "layout",
    "javascript",
    "nspr",
    "qa",
    "browser",
    "networking",
    "internationalization",
    "ui",
    "general",
    "other",
    "unknown",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def should_start_record(lines: List[str], idx: int) -> bool:
    line = lines[idx]
    match = CREATED_SPLIT_RE.match(line)
    if not match:
        return False

    indent_length = len(match.group("indent"))
    if indent_length > 1:
        return False

    previous_window = " ".join(
        ln.strip().lower() for ln in lines[max(0, idx - 2) : idx] if ln.strip()
    )
    if "in reply to comment" in previous_window:
        return False
    if "created an attachment" in previous_window:
        return False

    return True


def split_into_records(lines: List[str]) -> List[List[str]]:
    records: List[List[str]] = []
    current: List[str] = []
    started = False

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        if should_start_record(lines, idx):
            if current:
                records.append(current)
            current = [line]
            started = True
            continue

        if started:
            current.append(line)

    if current:
        records.append(current)

    return records


def extract_created_raw(record_lines: List[str]) -> str:
    for line in record_lines:
        match = CREATED_LINE_RE.match(line)
        if match:
            return normalize_whitespace(match.group("created"))
    return ""


def build_temporal_features(created_raw: str) -> Dict[str, object]:
    temporal: Dict[str, object] = {"created_raw": created_raw}
    if not created_raw:
        return temporal

    match = TEMPORAL_RE.search(created_raw)
    if not match:
        return temporal

    groups = match.groupdict()
    month_token = groups["month"]

    try:
        month_number = datetime.strptime(month_token[:3], "%b").month
        hour_12 = int(groups["hour"]) % 12
        hour_24 = hour_12 + (12 if groups["ampm"].upper() == "PM" else 0)
        dt = datetime(
            year=int(groups["year"]),
            month=month_number,
            day=int(groups["day"]),
            hour=hour_24,
            minute=int(groups["minute"]),
            second=int(groups["second"]),
        )
    except ValueError:
        return temporal

    temporal.update(
        {
            "created_iso": dt.isoformat(),
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "weekday": dt.strftime("%A"),
            "is_weekend": int(dt.weekday() >= 5),
            "timezone": groups.get("tz") or "",
        }
    )
    return temporal


def clean_non_metadata(lines: List[str]) -> str:
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(Created|Updated) by\b", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Additional Details\b", stripped, re.IGNORECASE):
            continue
        if re.match(r"^In reply to comment\b", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Created an attachment\b", stripped, re.IGNORECASE):
            continue
        cleaned.append(stripped)

    return normalize_whitespace(" ".join(cleaned))


def extract_textual_features(record_lines: List[str]) -> str:
    first_section: List[str] = []
    for line in record_lines:
        stripped = line.strip()
        if re.match(r"^Updated by\b", stripped, re.IGNORECASE):
            break
        first_section.append(line)

    text = clean_non_metadata(first_section)
    if len(text.split()) < 6:
        text = clean_non_metadata(record_lines)
    return text


def infer_priority(text: str) -> str:
    lower = text.lower()

    explicit = re.search(
        r"\b(?:priority|marking|set priority|upgrading priority to)\b[^a-z0-9]{0,10}\bp([1-5])\b",
        lower,
    )
    if explicit:
        return {
            "1": "critical",
            "2": "high",
            "3": "medium",
            "4": "low",
            "5": "low",
        }[explicit.group(1)]

    if re.search(r"\b(highest|critical|blocker)\b", lower):
        return "critical"
    if re.search(r"\bhigh\b", lower):
        return "high"
    if re.search(r"\b(medium|normal)\b", lower):
        return "medium"
    if re.search(r"\b(low|minor|trivial)\b", lower):
        return "low"
    return "unknown"


def infer_severity(text: str) -> str:
    lower = text.lower()

    numeric = re.search(r"\bseverity\s*([1-5])\b", lower)
    if numeric:
        return {
            "1": "critical",
            "2": "major",
            "3": "minor",
            "4": "trivial",
            "5": "trivial",
        }[numeric.group(1)]

    named = re.search(r"\bseverity\b[^a-z0-9]{0,10}(critical|major|minor|trivial)\b", lower)
    if named:
        return named.group(1)

    if "critical" in lower:
        return "critical"
    if "major" in lower:
        return "major"
    if "minor" in lower:
        return "minor"
    if "trivial" in lower:
        return "trivial"
    return "unknown"


def infer_issue_type(text: str) -> str:
    lower = text.lower()

    if re.search(r"\bregression\b|\bused to\b|\bno longer\b", lower):
        return "regression"
    if re.search(r"\bfeature request\b|\benhancement\b|\badd\b", lower):
        return "feature"
    if re.search(r"\bcompile\b|\bbuild\b|\bundefined symbol\b|\bmake\b.*\berror\b", lower):
        return "build"
    if re.search(r"\btoolbar\b|\bwindow\b|\bmenu\b|\bbutton\b|\blayout\b", lower):
        return "ui"
    if re.search(r"\bslow\b|\bperformance\b|\blatency\b", lower):
        return "performance"
    if re.search(r"\bsecurity\b|\bvulnerability\b|\bcve\b", lower):
        return "security"
    if re.search(r"\bcrash\b|\bbug\b|\berror\b|\bexception\b|\bassert\b", lower):
        return "bug"
    return "unknown"


def infer_component(text: str) -> str:
    lower = text.lower()

    component_keywords = {
        "layout": "layout",
        "javascript": "javascript",
        "nspr": "nspr",
        "qa": "qa",
        "browser": "browser",
        "network": "networking",
        "i18n": "internationalization",
        "internationalization": "internationalization",
        "toolbar": "ui",
        "ui": "ui",
        "general": "general",
    }

    for key, value in component_keywords.items():
        if f"component to {key}" in lower:
            return value
        if f"{key} component" in lower:
            return value

    for key, value in component_keywords.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return value

    return "unknown"


def one_hot(label: str, label_space: List[str]) -> List[int]:
    return [1 if label == token else 0 for token in label_space]


def build_categorical_features(record_lines: List[str], text: str) -> Dict[str, object]:
    source_text = "\n".join(record_lines) + "\n" + text
    priority = infer_priority(source_text)
    severity = infer_severity(source_text)
    issue_type = infer_issue_type(source_text)
    component = infer_component(source_text)

    vector = (
        one_hot(priority, PRIORITY_LABELS)
        + one_hot(severity, SEVERITY_LABELS)
        + one_hot(issue_type, TYPE_LABELS)
        + one_hot(component, COMPONENT_LABELS)
    )

    return {
        "priority": priority,
        "severity": severity,
        "issue_type": issue_type,
        "component": component,
        "vector": vector,
    }


def is_context_line(line: str) -> bool:
    lower = line.lower()

    keyword_match = re.search(
        (
            r"\b(stack trace|traceback|assert|exception|error|fatal|segmentation fault|"
            r"sigsegv|access violation|undefined symbol|crash|call stack)\b"
        ),
        lower,
    )
    if keyword_match:
        return True

    if re.search(r"\b0x[0-9a-f]{5,}\b", lower):
        return True

    if re.search(r"\b(?:cc|cxx|ld|gmake|make)\b.*\berror\b", lower):
        return True

    if re.search(r"\b[A-Za-z_][\w\-.]*\.(?:c|cc|cpp|h|hpp|java|js|py|cs|go|rb|php)\b", line):
        return True

    if re.match(
        r"^(if|for|while|switch|case|return|define|ifdef|ifndef|elif|endif|class|struct|void|int|char)\b",
        line,
        re.IGNORECASE,
    ):
        return True

    return False


def extract_contextual_features(record_lines: List[str]) -> str:
    snippets: List[str] = []
    i = 0

    while i < len(record_lines):
        stripped = record_lines[i].strip()
        if not stripped:
            i += 1
            continue

        lower = stripped.lower()
        if "stack trace" in lower or "traceback" in lower:
            for j in range(i, min(i + 12, len(record_lines))):
                candidate = record_lines[j].strip()
                if candidate:
                    snippets.append(normalize_whitespace(candidate))
            i += 12
            continue

        if is_context_line(stripped):
            snippets.append(normalize_whitespace(stripped))

        i += 1

    deduped: List[str] = []
    seen = set()
    for snippet in snippets:
        key = snippet.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(snippet)

    return " || ".join(deduped[:40])


def convert(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    records = split_into_records(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "report_id",
                "textual_features",
                "temporal_features",
                "categorical_features",
                "contextual_features",
            ],
        )
        writer.writeheader()

        for index, record_lines in enumerate(records, start=1):
            created_raw = extract_created_raw(record_lines)
            textual_features = extract_textual_features(record_lines)
            temporal_features = build_temporal_features(created_raw)
            categorical_features = build_categorical_features(record_lines, textual_features)
            contextual_features = extract_contextual_features(record_lines)

            writer.writerow(
                {
                    "report_id": index,
                    "textual_features": textual_features,
                    "temporal_features": json.dumps(temporal_features, ensure_ascii=True),
                    "categorical_features": json.dumps(categorical_features, ensure_ascii=True),
                    "contextual_features": contextual_features,
                }
            )

    print(f"Converted {len(records)} records to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert semi-structured bug text into feature-based CSV rows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset") / "corpus (fixsev).txt",
        help="Path to the semi-structured source text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset") / "corpus_features.csv",
        help="Path to the output CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
