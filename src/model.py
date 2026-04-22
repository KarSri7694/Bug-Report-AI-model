import argparse
import json
import logging
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from TorchCRF import CRF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


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

LABEL_COLUMN_CANDIDATES = ["label", "target", "is_duplicate", "duplicate", "bug_label"]
REPORT_LABELS = ["non_duplicate", "duplicate"]
REPORT_LABEL_TO_ID = {label: idx for idx, label in enumerate(REPORT_LABELS)}
STOP_WORDS = set(ENGLISH_STOP_WORDS)

DESCRIPTION_COLUMN_CANDIDATES = ["Description"]
PRIORITY_NAME_COLUMN_CANDIDATES = ["Priority Name", "Prioirty Name", "priority_name"]
PRIORITY_ID_COLUMN_CANDIDATES = ["Priority Id", "Priority ID", "Prioirity Id", "priority_id"]
BUG_CREATION_DATE_COLUMN_CANDIDATES = ["Bug Creation Date", "Bug creation Date", "created"]
BUG_ID_COLUMN_CANDIDATES = ["Bug Id", "Bug ID", "report_id", "Issue key"]
ASSIGNED_TO_COLUMN_CANDIDATES = ["Assigned To", "Assigned to", "assignee"]
FIX_VERSION_COLUMN_CANDIDATES = ["Fix Version", "Fix version"]
COMPONENTS_COLUMN_CANDIDATES = ["Components", "Component/s"]
RESOLUTION_COLUMN_CANDIDATES = ["Resolution"]

LOGGER = logging.getLogger(__name__)


def configure_logging(level_name: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure console/file logging for data preparation and training runs."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_text(text: str) -> List[str]:
    """Tokenize text and remove stopwords for the textual branch."""
    tokens = re.findall(r"[a-z0-9_]+", str(text).lower())
    return [token for token in tokens if token not in STOP_WORDS]


def normalize_text_for_training(text: str) -> str:
    """Lowercase text, remove punctuation, and remove stopwords."""
    return " ".join(tokenize_text(text))


def safe_json_load(value: object) -> Dict[str, object]:
    """Parse JSON-like text safely; returns an empty dict on parse errors."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def one_hot(label: str, labels: Sequence[str]) -> List[float]:
    """Convert a label into a dense one-hot list for fixed-width model input."""
    return [1.0 if label == token else 0.0 for token in labels]


def try_cast_vector(value: object) -> Optional[List[float]]:
    """Try converting a list-like value to a float vector."""
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def resolve_column_name(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Resolve a column name from candidate spellings in a case-insensitive way."""
    lowered = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        resolved = lowered.get(candidate.strip().lower())
        if resolved is not None:
            return resolved
    return None


def get_series_by_candidates(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Return a cleaned string series for the first matching column, else empty strings."""
    column = resolve_column_name(df, candidates)
    if column is None:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str).str.strip()


def extract_bug_id_number(raw: str) -> int:
    """Extract numeric suffix from IDs like LUCENE-123; returns 0 if absent."""
    text = str(raw).strip()
    if not text:
        return 0

    match = re.search(r"(\d+)$", text)
    if not match:
        match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0


def normalize_priority_from_fields(priority_name: str, priority_id: str) -> str:
    """Normalize priority using Priority Name and Priority Id values from issue CSVs."""
    name = str(priority_name).strip().lower()
    id_text = str(priority_id).strip()

    name_map = {
        "highest": "critical",
        "blocker": "critical",
        "critical": "critical",
        "high": "high",
        "major": "high",
        "medium": "medium",
        "normal": "medium",
        "low": "low",
        "minor": "low",
        "trivial": "low",
        "lowest": "low",
    }
    if name in name_map:
        return name_map[name]

    id_map = {1: "critical", 2: "high", 3: "medium", 4: "low", 5: "low"}

    numeric_name: Optional[int] = None
    if name.isdigit():
        numeric_name = int(name)
    if numeric_name in id_map:
        return id_map[numeric_name]

    try:
        numeric_id = int(float(id_text))
    except ValueError:
        numeric_id = -1
    if numeric_id in id_map:
        return id_map[numeric_id]

    return "unknown"


def build_first_seen_id_map(values: Sequence[str]) -> Dict[str, int]:
    """Create deterministic IDs by first appearance. Zero is reserved for empty values."""
    mapping: Dict[str, int] = {}
    next_id = 1

    for raw_value in values:
        normalized = str(raw_value).strip().lower()
        if not normalized:
            continue
        if normalized not in mapping:
            mapping[normalized] = next_id
            next_id += 1

    return mapping


def encode_context_value(raw_value: str, mapping: Dict[str, int]) -> int:
    normalized = str(raw_value).strip().lower()
    if not normalized:
        return 0
    return mapping.get(normalized, 0)


def has_requested_source_columns(df: pd.DataFrame) -> bool:
    """Check whether the dataframe can be converted to the 4-feature CSV format."""
    has_textual = resolve_column_name(df, DESCRIPTION_COLUMN_CANDIDATES) is not None
    has_categorical = (
        resolve_column_name(df, PRIORITY_NAME_COLUMN_CANDIDATES) is not None
        or resolve_column_name(df, PRIORITY_ID_COLUMN_CANDIDATES) is not None
    )
    has_temporal = (
        resolve_column_name(df, BUG_CREATION_DATE_COLUMN_CANDIDATES) is not None
        and resolve_column_name(df, BUG_ID_COLUMN_CANDIDATES) is not None
    )
    has_contextual = any(
        resolve_column_name(df, candidates) is not None
        for candidates in [
            ASSIGNED_TO_COLUMN_CANDIDATES,
            FIX_VERSION_COLUMN_CANDIDATES,
            COMPONENTS_COLUMN_CANDIDATES,
        ]
    )
    return has_textual and has_categorical and has_temporal and has_contextual


def build_four_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 4-feature dataset from tabular issue fields.

    Mapping used:
    - textual_features: Description
    - categorical_features: Priority Name, Priority Id
    - temporal_features: Bug Creation Date, Bug Id
    - contextual_features: Assigned To, Fix Version, Components
    """
    bug_ids = get_series_by_candidates(df, BUG_ID_COLUMN_CANDIDATES)
    descriptions = get_series_by_candidates(df, DESCRIPTION_COLUMN_CANDIDATES)
    priority_names = get_series_by_candidates(df, PRIORITY_NAME_COLUMN_CANDIDATES)
    priority_ids = get_series_by_candidates(df, PRIORITY_ID_COLUMN_CANDIDATES)
    bug_creation_dates = get_series_by_candidates(df, BUG_CREATION_DATE_COLUMN_CANDIDATES)
    assigned_to = get_series_by_candidates(df, ASSIGNED_TO_COLUMN_CANDIDATES)
    fix_versions = get_series_by_candidates(df, FIX_VERSION_COLUMN_CANDIDATES)
    components = get_series_by_candidates(df, COMPONENTS_COLUMN_CANDIDATES)
    resolutions = get_series_by_candidates(df, RESOLUTION_COLUMN_CANDIDATES)

    component_id_map = build_first_seen_id_map(components.tolist())
    assigned_to_id_map = build_first_seen_id_map(assigned_to.tolist())
    fix_version_id_map = build_first_seen_id_map(fix_versions.tolist())

    lucene1_datetime: Optional[datetime] = None
    for index in range(len(df)):
        bug_id_value = str(bug_ids.iat[index]).strip().lower()
        if bug_id_value == "lucene-1":
            lucene1_datetime = parse_datetime(str(bug_creation_dates.iat[index]))
            break

    if lucene1_datetime is None:
        for raw_created in bug_creation_dates.tolist():
            parsed = parse_datetime(str(raw_created))
            if parsed is not None:
                lucene1_datetime = parsed
                break

    report_ids: List[str] = []
    categorical_payloads: List[str] = []
    temporal_payloads: List[str] = []
    contextual_features: List[str] = []

    for index in range(len(df)):
        bug_id_value = bug_ids.iat[index]
        report_ids.append(bug_id_value if bug_id_value else f"report-{index + 1}")

        priority_label = normalize_priority_from_fields(priority_names.iat[index], priority_ids.iat[index])
        categorical_payload = {
            "priority_name": priority_names.iat[index],
            "priority_id": priority_ids.iat[index],
            "priority": priority_label,
            "vector": (
                one_hot(priority_label, PRIORITY_LABELS)
                + one_hot("unknown", SEVERITY_LABELS)
                + one_hot("unknown", TYPE_LABELS)
                + one_hot("unknown", COMPONENT_LABELS)
            ),
        }
        categorical_payloads.append(json.dumps(categorical_payload, ensure_ascii=True))

        bug_id_number = extract_bug_id_number(bug_id_value)
        created_raw = bug_creation_dates.iat[index]
        created_dt = parse_datetime(str(created_raw))
        epoch_start = datetime(1970, 1, 1)
        days_since_epoch = 0.0
        gap_from_lucene1_hours = 0.0
        if created_dt is not None:
            days_since_epoch = float((created_dt - epoch_start).days)
            if lucene1_datetime is not None:
                gap_from_lucene1_hours = float((created_dt - lucene1_datetime).total_seconds() / 3600.0)

        temporal_payload = {
            "created_raw": created_raw,
            "bug_id": bug_id_value,
            "bug_id_numeric": bug_id_number,
            "created_days_since_epoch": days_since_epoch,
            "gap_from_lucene1_hours": gap_from_lucene1_hours,
        }
        temporal_payloads.append(json.dumps(temporal_payload, ensure_ascii=True))

        assigned_to_value = assigned_to.iat[index]
        fix_version_value = fix_versions.iat[index]
        component_value = components.iat[index]

        contextual_payload = {
            "assigned_to": assigned_to_value,
            "assigned_to_id": encode_context_value(assigned_to_value, assigned_to_id_map),
            "fix_version": fix_version_value,
            "fix_version_id": encode_context_value(fix_version_value, fix_version_id_map),
            "components": component_value,
            "components_id": encode_context_value(component_value, component_id_map),
        }
        contextual_features.append(
            json.dumps(contextual_payload, ensure_ascii=True)
        )

    feature_df = pd.DataFrame(
        {
            "report_id": report_ids,
            "textual_features": descriptions,
            "categorical_features": categorical_payloads,
            "temporal_features": temporal_payloads,
            "contextual_features": contextual_features,
        }
    )

    if (resolutions.str.len() > 0).any():
        feature_df["Resolution"] = resolutions

    return feature_df


def infer_priority(text: str) -> str:
    lower = text.lower()
    explicit = re.search(
        r"\b(?:priority|set priority|upgrading priority to)\b[^a-z0-9]{0,10}\bp([1-5])\b",
        lower,
    )
    if explicit:
        return {"1": "critical", "2": "high", "3": "medium", "4": "low", "5": "low"}[explicit.group(1)]
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
        if f"component to {key}" in lower or f"{key} component" in lower:
            return value
    for key, value in component_keywords.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return value
    return "unknown"


def infer_weak_report_label(text: str) -> int:
    """
    Weak label fallback used when no explicit target column is available.
    Returns class IDs for Duplicate/Non-duplicate.
    """
    patterns = [
        r"\bduplicate of\b",
        r"\bmarked as a duplicate\b",
        r"\bmarking dup of\b",
        r"\bthis bug has been marked as a duplicate\b",
        r"\bdup of\b",
        r"\bresolved as duplicate\b",
    ]
    lower = text.lower()
    if any(re.search(pattern, lower) for pattern in patterns):
        return REPORT_LABEL_TO_ID["duplicate"]
    return REPORT_LABEL_TO_ID["non_duplicate"]


def parse_datetime(raw: str) -> Optional[datetime]:
    """Parse date-time from multiple known bug-report formats."""
    if not raw:
        return None

    # Format from converted corpus_features.csv (e.g., Tuesday April 7 1998 10 57 25 AM PDT)
    clean = re.sub(r"\s+", " ", raw).strip()
    pattern = re.compile(
        (
            r"(?P<weekday>[A-Za-z]+)\s+"
            r"(?P<month>[A-Za-z]+)\s+"
            r"(?P<day>\d{1,2})\s+"
            r"(?P<year>\d{4})\s+"
            r"(?P<hour>\d{1,2})\s+"
            r"(?P<minute>\d{1,2})\s+"
            r"(?P<second>\d{1,2})\s+"
            r"(?P<ampm>AM|PM)"
        ),
        re.IGNORECASE,
    )
    match = pattern.search(clean)
    if match:
        g = match.groupdict()
        month_num = datetime.strptime(g["month"][:3], "%b").month
        hour12 = int(g["hour"]) % 12
        hour24 = hour12 + (12 if g["ampm"].upper() == "PM" else 0)
        return datetime(
            year=int(g["year"]),
            month=month_num,
            day=int(g["day"]),
            hour=hour24,
            minute=int(g["minute"]),
            second=int(g["second"]),
        )

    # Format from Jira-like exports (e.g., 29/May/2023 6:43 AM)
    for fmt in [
        "%d/%b/%Y %I:%M %p",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]:
        try:
            parsed = datetime.strptime(clean, fmt)
            if parsed.tzinfo is not None:
                parsed = parsed.replace(tzinfo=None)
            return parsed
        except ValueError:
            continue
    return None


def build_temporal_vector(row: pd.Series) -> List[float]:
    """
    Build temporal vector with date math:
    1) days since Unix epoch
    2) gap from LUCENE-1 creation time in hours
    """
    precomputed_vector = try_cast_vector(row.get("temporal_vector", None))
    if precomputed_vector is not None:
        return precomputed_vector

    payload = safe_json_load(row.get("temporal_features", ""))
    if payload:
        days_value = payload.get("created_days_since_epoch")
        gap_value = payload.get("gap_from_lucene1_hours")
        if days_value is not None and gap_value is not None:
            try:
                return [float(days_value), float(gap_value)]
            except (TypeError, ValueError):
                pass

    created_raw = str(payload.get("created_raw", "")) if payload else ""
    if not created_raw and "Bug Creation Date" in row:
        created_raw = str(row.get("Bug Creation Date", ""))
    if not created_raw and "Created" in row:
        created_raw = str(row.get("Created", ""))

    created_dt = parse_datetime(created_raw)
    if created_dt is None:
        return [0.0, 0.0]

    epoch_start = datetime(1970, 1, 1)
    days_since_epoch = float((created_dt - epoch_start).days)

    lucene1_raw = str(payload.get("lucene1_created_raw", "")) if payload else ""
    lucene1_dt = parse_datetime(lucene1_raw) if lucene1_raw else None
    gap_hours = float((created_dt - lucene1_dt).total_seconds() / 3600.0) if lucene1_dt else 0.0
    return [days_since_epoch, gap_hours]


def build_categorical_vector(row: pd.Series, fallback_text: str) -> List[float]:
    """Build a fixed-width categorical vector from JSON features or fallback heuristics."""
    precomputed_vector = try_cast_vector(row.get("categorical_vector", None))
    if precomputed_vector is not None:
        return precomputed_vector

    payload = safe_json_load(row.get("categorical_features", ""))
    vector_obj = payload.get("vector") if payload else None
    if isinstance(vector_obj, list):
        return [float(value) for value in vector_obj]

    priority = str(payload.get("priority", "")) if payload else ""
    priority_name = str(payload.get("priority_name", "")) if payload else ""
    priority_id = str(payload.get("priority_id", "")) if payload else ""
    severity = str(payload.get("severity", "")) if payload else ""
    issue_type = str(payload.get("issue_type", "")) if payload else ""
    component = str(payload.get("component", "")) if payload else ""

    if not priority and (priority_name or priority_id):
        priority = normalize_priority_from_fields(priority_name, priority_id)

    # Fall back to paper-inspired categorical extraction when structured fields are missing.
    merged = " ".join(
        [
            fallback_text,
            str(row.get("Priority", "")),
            str(row.get("Severity", "")),
            str(row.get("Issue Type", "")),
            str(row.get("Component/s", "")),
        ]
    )
    if not priority:
        priority = infer_priority(merged)
    if not severity:
        severity = infer_severity(merged)
    if not issue_type:
        issue_type = infer_issue_type(merged)
    if not component:
        component = infer_component(merged)

    vector = (
        one_hot(priority, PRIORITY_LABELS)
        + one_hot(severity, SEVERITY_LABELS)
        + one_hot(issue_type, TYPE_LABELS)
        + one_hot(component, COMPONENT_LABELS)
    )
    return [float(v) for v in vector]


def build_contextual_vector(row: pd.Series) -> List[float]:
    """
    Context branch with categorical ID encodings:
    [components_id, assigned_to_id, fix_version_id]
    """
    precomputed_vector = try_cast_vector(row.get("contextual_vector", None))
    if precomputed_vector is not None:
        return precomputed_vector

    payload = safe_json_load(row.get("contextual_features", ""))
    if payload:
        try:
            components_id = float(payload.get("components_id", payload.get("component_id", 0)) or 0)
            assigned_to_id = float(payload.get("assigned_to_id", 0) or 0)
            fix_version_id = float(payload.get("fix_version_id", 0) or 0)
            return [components_id, assigned_to_id, fix_version_id]
        except (TypeError, ValueError):
            pass

    contextual_text = str(row.get("contextual_features", "")).strip().lower()
    if contextual_text:
        component_present = bool(re.search(r"components\s*:\s*[^|\s]", contextual_text))
        assigned_present = bool(re.search(r"assigned_to\s*:\s*[^|\s]", contextual_text))
        fix_present = bool(re.search(r"fix_version\s*:\s*[^|\s]", contextual_text))
        return [float(component_present), float(assigned_present), float(fix_present)]

    component_raw = str(row.get("Components", row.get("Component/s", ""))).strip()
    assigned_raw = str(row.get("Assigned To", "")).strip()
    fix_raw = str(row.get("Fix Version", "")).strip()
    return [float(bool(component_raw)), float(bool(assigned_raw)), float(bool(fix_raw))]


def build_contextual_plain_text(row: pd.Series) -> str:
    """Build human-readable contextual text without raw JSON syntax."""
    payload = safe_json_load(row.get("contextual_features", ""))
    if payload:
        parts: List[str] = []
        for key in ["assigned_to", "fix_version", "components"]:
            value = str(payload.get(key, "")).strip()
            if value:
                parts.append(value)
        return " ".join(parts)

    contextual_text = str(row.get("contextual_features", "")).strip()
    if contextual_text:
        return contextual_text

    component_raw = str(row.get("Components", row.get("Component/s", ""))).strip()
    assigned_raw = str(row.get("Assigned To", "")).strip()
    fix_raw = str(row.get("Fix Version", "")).strip()
    return " ".join(part for part in [assigned_raw, fix_raw, component_raw] if part)


@dataclass
class BugExample:
    """Container for a single bug report sample with all feature branches."""

    text: str
    temporal_vector: List[float]
    categorical_vector: List[float]
    contextual_vector: List[float]
    label: int


class Vocabulary:
    """Vocabulary for text-to-id conversion in the embedding branch."""

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.id_to_token: List[str] = [self.PAD, self.UNK]

    def build(self, texts: Sequence[str], max_size: int) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize_text(text))

        for token, _ in counter.most_common(max_size - len(self.token_to_id)):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.id_to_token)
                self.id_to_token.append(token)

    def encode(self, text: str, max_seq_len: int) -> Dict[str, torch.Tensor]:
        tokens = tokenize_text(text)
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens]

        # Truncate or pad to fixed sequence length for efficient batching.
        token_ids = token_ids[:max_seq_len]
        attention_mask = [1] * len(token_ids)
        if len(token_ids) < max_seq_len:
            pad_count = max_seq_len - len(token_ids)
            token_ids.extend([self.token_to_id[self.PAD]] * pad_count)
            attention_mask.extend([0] * pad_count)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }

    @property
    def size(self) -> int:
        return len(self.id_to_token)


def load_modeling_dataframe(data_path: Path) -> pd.DataFrame:
    """Load CSV and adapt columns so ADLM can be trained from multiple formats."""
    df = pd.read_csv(data_path)
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    required_feature_columns = {
        "textual_features",
        "categorical_features",
        "temporal_features",
        "contextual_features",
    }
    if not required_feature_columns.issubset(set(df.columns)) and has_requested_source_columns(df):
        feature_df = build_four_feature_dataframe(df)
        output_path = data_path.with_name(f"{data_path.stem}_4features.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path, index=False)
        LOGGER.info("Saved 4-feature CSV to: %s", output_path)
        df = feature_df

    # If the file is the converted corpus output, these columns already exist.
    if "textual_features" not in df.columns:
        summary = df.get("Summary", pd.Series([""] * len(df))).fillna("")
        description = df.get("Description", pd.Series([""] * len(df))).fillna("")
        df["textual_features"] = (summary.astype(str) + " " + description.astype(str)).str.strip()

    if "contextual_features" not in df.columns:
        df["contextual_features"] = df.get("Description", pd.Series([""] * len(df))).fillna("")

    # Enforce punctuation/stopword cleanup at data preparation time.
    df["textual_features"] = df["textual_features"].fillna("").astype(str).map(normalize_text_for_training)

    # Materialize vectors once so training uses parsed numeric inputs, not raw JSON text from CSV.
    df["contextual_text_plain"] = df.apply(build_contextual_plain_text, axis=1)
    df["contextual_text_plain"] = df["contextual_text_plain"].fillna("").astype(str).map(normalize_text_for_training)
    df["temporal_vector"] = df.apply(build_temporal_vector, axis=1)
    df["contextual_vector"] = df.apply(build_contextual_vector, axis=1)
    df["categorical_vector"] = df.apply(
        lambda row: build_categorical_vector(
            row,
            str(row.get("textual_features", "")).strip() + " " + str(row.get("contextual_text_plain", "")).strip(),
        ),
        axis=1,
    )

    return df


def resolve_training_data_paths(
    data_paths: Optional[Sequence[Path]],
    data_path: Optional[Path],
) -> List[Path]:
    """Resolve input paths from explicit CLI dataset args."""
    if data_paths:
        return [Path(path) for path in data_paths]
    if data_path is not None:
        return [Path(data_path)]
    raise ValueError(
        "No dataset path provided. Use --data_path <csv> or --data_paths <csv1> [<csv2> ...]."
    )


def derive_binary_label(row: pd.Series, label_column: Optional[str]) -> int:
    """Create a binary label from explicit labels when available, else weak supervision."""
    text = str(row.get("textual_features", "")).strip()
    contextual_text = str(row.get("contextual_text_plain", "")).strip()
    if not contextual_text:
        contextual_text = str(row.get("contextual_features", "")).strip()
    merged_text = (text + " " + contextual_text).strip()

    if label_column:
        explicit_label = normalize_label(row.get(label_column), merged_text)
        if explicit_label is not None:
            return explicit_label

    return infer_weak_report_label(merged_text)


def balance_binary_dataframe(df: pd.DataFrame, label_column: str, seed: int) -> pd.DataFrame:
    """Down-sample to equal Duplicate/Non-duplicate counts."""
    counts = df[label_column].value_counts()
    if len(counts) < len(REPORT_LABELS):
        raise ValueError("Both Duplicate and Non-duplicate classes are required for balancing.")

    target_size = int(counts.min())
    balanced_parts: List[pd.DataFrame] = []
    for class_id in range(len(REPORT_LABELS)):
        class_rows = df[df[label_column] == class_id]
        if class_rows.empty:
            raise ValueError("Both Duplicate and Non-duplicate classes are required for balancing.")
        balanced_parts.append(class_rows.sample(n=target_size, random_state=seed))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def prepare_training_dataframe(
    data_paths: Sequence[Path],
    explicit_label_column: Optional[str],
    seed: int,
) -> pd.DataFrame:
    """Load, normalize, and balance one or more datasets into a binary training frame."""
    prepared_frames: List[pd.DataFrame] = []

    for data_path in data_paths:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        frame = load_modeling_dataframe(data_path)
        resolved_label_column = detect_label_column(frame, explicit_label_column)
        frame = frame.copy()
        frame["dataset_source"] = data_path.stem
        frame["binary_label"] = frame.apply(
            lambda row: derive_binary_label(row, resolved_label_column),
            axis=1,
        )

        LOGGER.info(
            "Prepared dataset: %s | rows=%d | label_source=%s",
            data_path,
            len(frame),
            resolved_label_column if resolved_label_column else "weak_supervision",
        )
        prepared_frames.append(frame)

    if not prepared_frames:
        raise ValueError("No datasets were prepared.")

    combined = pd.concat(prepared_frames, ignore_index=True)
    combined = combined[combined["textual_features"].str.len() > 0].copy()
    combined["binary_label"] = combined["binary_label"].astype(int)

    balanced = balance_binary_dataframe(combined, label_column="binary_label", seed=seed)
    counts = balanced["binary_label"].value_counts().to_dict()
    LOGGER.info(
        "Balanced samples: %s",
        {
            "non_duplicate": int(counts.get(REPORT_LABEL_TO_ID["non_duplicate"], 0)),
            "duplicate": int(counts.get(REPORT_LABEL_TO_ID["duplicate"], 0)),
        },
    )
    return balanced


def detect_label_column(df: pd.DataFrame, explicit_label_column: Optional[str]) -> Optional[str]:
    """Resolve the supervision column if it exists in the dataset."""
    lowered = {str(column).strip().lower(): str(column) for column in df.columns}
    if explicit_label_column:
        if explicit_label_column in df.columns:
            return explicit_label_column
        explicit_match = lowered.get(explicit_label_column.strip().lower())
        if explicit_match is not None:
            return explicit_match

    for candidate in LABEL_COLUMN_CANDIDATES + ["resolution"]:
        match = lowered.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def normalize_label(value: object, fallback_text: str = "") -> Optional[int]:
    """Normalize labels into class IDs for Duplicate/Non-duplicate."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    if isinstance(value, (int, float)):
        numeric_value = int(float(value))
        if numeric_value == 1:
            return REPORT_LABEL_TO_ID["duplicate"]
        if numeric_value in (0, 2):
            return REPORT_LABEL_TO_ID["non_duplicate"]
        return None

    text = str(value).strip().lower()
    duplicate_tokens = {
        "duplicate",
        "dup",
        "is duplicate",
        "yes",
        "true",
    }
    non_duplicate_tokens = {
        "non duplicate",
        "not duplicate",
        "nonduplicate",
        "bug",
        "enhancement",
        "feature",
        "feature request",
        "improvement",
        "bug",
        "fixed",
        "wont fix",
        "won't fix",
        "incomplete",
        "cannot reproduce",
        "invalid",
        "not a problem",
        "works for me",
        "done",
        "closed",
        "resolved",
        "no",
        "false",
    }

    normalized_text = re.sub(r"[^a-z0-9]+", " ", text).strip()

    if "not duplicate" in normalized_text or "non duplicate" in normalized_text:
        return REPORT_LABEL_TO_ID["non_duplicate"]

    if text in duplicate_tokens or normalized_text in duplicate_tokens:
        return REPORT_LABEL_TO_ID["duplicate"]
    if text in non_duplicate_tokens or normalized_text in non_duplicate_tokens:
        return REPORT_LABEL_TO_ID["non_duplicate"]
    if "duplicate" in normalized_text:
        return REPORT_LABEL_TO_ID["duplicate"]
    if "enhancement" in normalized_text or "feature" in normalized_text:
        return REPORT_LABEL_TO_ID["non_duplicate"]

    if fallback_text:
        return infer_weak_report_label(fallback_text)

    return None


def build_examples(df: pd.DataFrame, label_column: Optional[str]) -> List[BugExample]:
    """
    Convert dataframe rows into model-ready examples.
    If no target column exists, weak labels are inferred from duplicate keywords.
    """
    examples: List[BugExample] = []

    for _, row in df.iterrows():
        text = str(row.get("textual_features", "")).strip()
        if not text:
            continue

        contextual_text = str(row.get("contextual_text_plain", "")).strip()
        if not contextual_text:
            contextual_text = str(row.get("contextual_features", "")).strip()
        merged_text = text + " " + contextual_text
        label: Optional[int] = None
        if label_column:
            label = normalize_label(row.get(label_column), merged_text)

        if label is None:
            # Weak supervision fallback keeps the training script runnable on unlabeled corpora.
            label = infer_weak_report_label(merged_text)

        examples.append(
            BugExample(
                text=text,
                temporal_vector=build_temporal_vector(row),
                categorical_vector=build_categorical_vector(row, merged_text),
                contextual_vector=build_contextual_vector(row),
                label=int(label),
            )
        )

    return examples


class BugReportDataset(Dataset):
    """PyTorch dataset that yields all ADLM branches per sample."""

    def __init__(self, examples: Sequence[BugExample], vocabulary: Vocabulary, max_seq_len: int) -> None:
        self.examples = list(examples)
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len

        # Ensure fixed vector width across the full split.
        self.temporal_dim = len(self.examples[0].temporal_vector) if self.examples else 0
        self.categorical_dim = len(self.examples[0].categorical_vector) if self.examples else 0
        self.contextual_dim = len(self.examples[0].contextual_vector) if self.examples else 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.vocabulary.encode(example.text, self.max_seq_len)

        label = torch.tensor(example.label, dtype=torch.long)

        # CRF expects token-level labels; we broadcast the report label over non-padding tokens.
        tag_ids = torch.full((self.max_seq_len,), fill_value=example.label, dtype=torch.long)
        tag_ids = tag_ids * encoded["attention_mask"].long()

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "temporal_features": torch.tensor(example.temporal_vector, dtype=torch.float32),
            "categorical_features": torch.tensor(example.categorical_vector, dtype=torch.float32),
            "contextual_features": torch.tensor(example.contextual_vector, dtype=torch.float32),
            "label": label,
            "tag_ids": tag_ids,
        }


class ADLMBiLSTMCRF(nn.Module):
    """
    ADLM-inspired architecture from the paper:
    text embedding + BiLSTM + fused temporal/categorical/contextual features + CRF.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        dropout: float,
        temporal_dim: int,
        categorical_dim: int,
        contextual_dim: int,
        num_labels: int = len(REPORT_LABELS),
        crf_regularization: float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.crf_regularization = crf_regularization
        self.bidirectional = False
        self.text_output_dim = hidden_size * (2 if self.bidirectional else 1)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.text_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        side_input_dim = temporal_dim + categorical_dim + contextual_dim
        # Side-channel MLP aligns feature scales before fusion with token states.
        self.side_encoder = nn.Sequential(
            nn.Linear(max(1, side_input_dim), self.text_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.text_output_dim, self.text_output_dim),
            nn.ReLU(),
        )

        # Fusion head maps [text_state || side_state] -> token emissions for CRF.
        self.emission_head = nn.Sequential(
            nn.Linear(self.text_output_dim * 2, self.text_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.text_output_dim, num_labels),
        )

        self.crf = CRF(num_labels, pad_idx=None, use_gpu=torch.cuda.is_available())

    def _crf_l2_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for parameter in self.crf.parameters():
            penalty = penalty + torch.sum(parameter ** 2)
        return penalty

    @staticmethod
    def token_tags_to_report_labels(tag_sequences: List[List[int]]) -> List[int]:
        """Collapse token predictions into one report-level class decision."""
        predictions: List[int] = []
        for tags in tag_sequences:
            if not tags:
                predictions.append(REPORT_LABEL_TO_ID["non_duplicate"])
                continue
            counts = Counter(tags)
            prediction = max(counts.items(), key=lambda item: item[1])[0]
            predictions.append(int(prediction))
        return predictions

    @staticmethod
    def class_probabilities_to_report_labels(class_probabilities: torch.Tensor) -> List[int]:
        """Convert per-class probabilities into report-level class IDs."""
        if class_probabilities.ndim != 2:
            return []
        return class_probabilities.argmax(dim=1).detach().cpu().tolist()

    def _compute_class_probabilities(self, emissions: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Estimate P(class) with CRF log-likelihood for constant-label report sequences."""
        log_likelihoods: List[torch.Tensor] = []
        for class_index in range(self.num_labels):
            tags = torch.full(
                emissions.shape[:2],
                fill_value=class_index,
                dtype=torch.long,
                device=emissions.device,
            )
            class_log_likelihood = self.crf(emissions, tags, mask)
            log_likelihoods.append(class_log_likelihood.unsqueeze(1))

        stacked = torch.cat(log_likelihoods, dim=1)
        return torch.softmax(stacked, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temporal_features: torch.Tensor,
        categorical_features: torch.Tensor,
        contextual_features: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        # 1) Text branch from paper equation ht = LSTM(xt, ht-1).
        embeddings = self.embedding(input_ids)
        lstm_output, _ = self.text_encoder(embeddings)
        lstm_output = self.dropout(lstm_output)

        # 2) Concatenate non-text features and project to match text hidden width.
        side_features = torch.cat([temporal_features, categorical_features, contextual_features], dim=-1)
        if side_features.shape[-1] == 0:
            side_features = torch.zeros((input_ids.size(0), 1), device=input_ids.device)

        side_representation = self.side_encoder(side_features)
        side_representation = side_representation.unsqueeze(1).expand(-1, input_ids.size(1), -1)

        # 3) Fuse and score each token for CRF decoding.
        fused_representation = torch.cat([lstm_output, side_representation], dim=-1)
        emissions = self.emission_head(fused_representation)

        crf_mask = cast(torch.BoolTensor, attention_mask.bool())
        decoded_tags = self.crf.viterbi_decode(emissions, crf_mask)
        class_probabilities = self._compute_class_probabilities(emissions, crf_mask)

        output: Dict[str, object] = {
            "decoded_tags": decoded_tags,
            "emissions": emissions,
            "class_probabilities": class_probabilities,
        }

        if tags is not None:
            # TorchCRF returns log-likelihood; we minimize negative log-likelihood.
            log_likelihood = self.crf(emissions, tags.long(), crf_mask)
            loss = -log_likelihood.mean()

            # Optional CRF regularization from the paper's optimizer search space.
            if self.crf_regularization > 0.0:
                loss = loss + self.crf_regularization * self._crf_l2_penalty()
            output["loss"] = loss

        return output


@dataclass
class TrainConfig:
    data_paths: List[Path]
    prepared_data_out: Path
    start_training: bool
    log_level: str
    log_file: Optional[Path]
    model_out: Path
    label_column: Optional[str]
    max_vocab_size: int
    max_seq_len: int
    embedding_dim: int
    hidden_size: int
    dropout: float
    learning_rate: float
    crf_regularization: float
    batch_size: int
    epochs: int
    val_size: float
    seed: int
    use_dragonfly: bool
    dragonfly_population: int
    dragonfly_iterations: int
    dragonfly_inner_epochs: int


@dataclass
class DataArtifacts:
    vocabulary: Vocabulary
    train_dataset: BugReportDataset
    val_dataset: BugReportDataset
    train_loader: DataLoader
    val_loader: DataLoader


@dataclass
class HyperCandidate:
    learning_rate: float
    hidden_size: int
    dropout: float
    crf_regularization: float


class DragonflyHyperOptimizer:
    """
    Lightweight Dragonfly-inspired optimizer for ADLM hyperparameters.
    This mirrors the paper's idea: optimize LR, hidden units, dropout, CRF regularization.
    """

    def __init__(self, population: int, iterations: int, seed: int) -> None:
        self.population = population
        self.iterations = iterations
        self.random = np.random.default_rng(seed)
        self.bounds = {
            "learning_rate": (1e-4, 5e-3),
            "hidden_size": (48.0, 256.0),
            "dropout": (0.1, 0.6),
            "crf_regularization": (1e-6, 5e-3),
        }

    def _clip(self, vector: np.ndarray) -> np.ndarray:
        clipped = vector.copy()
        keys = list(self.bounds.keys())
        for idx, key in enumerate(keys):
            lo, hi = self.bounds[key]
            clipped[idx] = np.clip(clipped[idx], lo, hi)
        return clipped

    def _to_candidate(self, vector: np.ndarray) -> HyperCandidate:
        return HyperCandidate(
            learning_rate=float(vector[0]),
            hidden_size=int(round(float(vector[1]) / 8.0) * 8),
            dropout=float(vector[2]),
            crf_regularization=float(vector[3]),
        )

    def _sample_population(self) -> np.ndarray:
        vectors = []
        for _ in range(self.population):
            vectors.append(
                np.array(
                    [
                        self.random.uniform(*self.bounds["learning_rate"]),
                        self.random.uniform(*self.bounds["hidden_size"]),
                        self.random.uniform(*self.bounds["dropout"]),
                        self.random.uniform(*self.bounds["crf_regularization"]),
                    ],
                    dtype=np.float64,
                )
            )
        return np.stack(vectors)

    def optimize(self, objective: Callable[[HyperCandidate], float]) -> HyperCandidate:
        population = self._sample_population()
        delta = np.zeros_like(population)

        fitness = np.array([objective(self._to_candidate(v)) for v in population], dtype=np.float64)

        for _ in range(self.iterations):
            best_index = int(np.argmax(fitness))
            worst_index = int(np.argmin(fitness))
            best_position = population[best_index]
            worst_position = population[worst_index]
            mean_position = np.mean(population, axis=0)
            mean_velocity = np.mean(delta, axis=0)

            for i in range(self.population):
                # Simplified Dragonfly forces: separation, alignment, cohesion, food, enemy.
                separation = -(population[i] - mean_position)
                alignment = mean_velocity
                cohesion = mean_position - population[i]
                food = best_position - population[i]
                enemy = population[i] - worst_position
                noise = self.random.normal(0.0, 0.01, size=population.shape[1])

                delta[i] = (
                    0.35 * delta[i]
                    + 0.10 * separation
                    + 0.10 * alignment
                    + 0.10 * cohesion
                    + 0.40 * food
                    + 0.20 * enemy
                    + noise
                )
                population[i] = self._clip(population[i] + delta[i])

            fitness = np.array([objective(self._to_candidate(v)) for v in population], dtype=np.float64)

        best_index = int(np.argmax(fitness))
        return self._to_candidate(population[best_index])


def compute_metrics(targets: List[int], predictions: List[int]) -> Dict[str, float]:
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(REPORT_LABELS))),
        average="macro",
        zero_division=0,
    )
    kappa = cohen_kappa_score(targets, predictions)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "kappa": float(kappa),
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device)
    return moved


def train_one_epoch(
    model: ADLMBiLSTMCRF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            temporal_features=batch["temporal_features"],
            categorical_features=batch["categorical_features"],
            contextual_features=batch["contextual_features"],
            tags=batch["tag_ids"],
        )
        loss = output["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: ADLMBiLSTMCRF,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    targets: List[int] = []
    predictions: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                temporal_features=batch["temporal_features"],
                categorical_features=batch["categorical_features"],
                contextual_features=batch["contextual_features"],
                tags=batch["tag_ids"],
            )

            if "loss" in output:
                losses.append(float(output["loss"].detach().cpu().item()))

            class_probabilities = cast(torch.Tensor, output["class_probabilities"])
            batch_predictions = ADLMBiLSTMCRF.class_probabilities_to_report_labels(class_probabilities)
            batch_targets = batch["label"].detach().cpu().tolist()

            predictions.extend(batch_predictions)
            targets.extend(batch_targets)

    metrics = compute_metrics(targets, predictions)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def split_examples(examples: Sequence[BugExample], val_size: float, seed: int) -> List[List[BugExample]]:
    labels = [example.label for example in examples]

    # Stratify only when both classes are present with enough samples.
    stratify = None
    label_set = set(labels)
    if len(label_set) > 1:
        counts = Counter(labels)
        if min(counts.values()) >= 2:
            stratify = labels

    train_examples, val_examples = train_test_split(
        list(examples),
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    return [train_examples, val_examples]


def build_dataloaders(
    train_examples: Sequence[BugExample],
    val_examples: Sequence[BugExample],
    max_vocab_size: int,
    max_seq_len: int,
    batch_size: int,
) -> DataArtifacts:
    vocabulary = Vocabulary()
    vocabulary.build([example.text for example in train_examples], max_size=max_vocab_size)

    train_dataset = BugReportDataset(train_examples, vocabulary=vocabulary, max_seq_len=max_seq_len)
    val_dataset = BugReportDataset(val_examples, vocabulary=vocabulary, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return DataArtifacts(
        vocabulary=vocabulary,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def run_training(config: TrainConfig) -> None:
    configure_logging(config.log_level, config.log_file)
    set_seed(config.seed)

    LOGGER.info("Using datasets: %s", [str(path) for path in config.data_paths])
    dataframe = prepare_training_dataframe(
        data_paths=config.data_paths,
        explicit_label_column=config.label_column,
        seed=config.seed,
    )

    config.prepared_data_out.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(config.prepared_data_out, index=False)
    LOGGER.info("Saved prepared balanced dataset to: %s", config.prepared_data_out)

    if not config.start_training:
        LOGGER.info("Preparation complete. Stopping before training. Pass --start_training to run training.")
        return

    label_column = "binary_label"
    examples = build_examples(dataframe, label_column=label_column)
    if len(examples) < 10:
        raise ValueError("Not enough valid training examples found in the dataset.")

    train_examples, val_examples = split_examples(examples, val_size=config.val_size, seed=config.seed)
    artifacts = build_dataloaders(
        train_examples=train_examples,
        val_examples=val_examples,
        max_vocab_size=config.max_vocab_size,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
    )

    train_dataset = artifacts.train_dataset
    train_loader = artifacts.train_loader
    val_loader = artifacts.val_loader
    vocabulary = artifacts.vocabulary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(candidate: Optional[HyperCandidate] = None) -> ADLMBiLSTMCRF:
        hidden_size = candidate.hidden_size if candidate else config.hidden_size
        dropout = candidate.dropout if candidate else config.dropout
        crf_reg = candidate.crf_regularization if candidate else config.crf_regularization

        model = ADLMBiLSTMCRF(
            vocab_size=vocabulary.size,
            embedding_dim=config.embedding_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            temporal_dim=train_dataset.temporal_dim,
            categorical_dim=train_dataset.categorical_dim,
            contextual_dim=train_dataset.contextual_dim,
            num_labels=len(REPORT_LABELS),
            crf_regularization=crf_reg,
        )
        return model.to(device)

    chosen_lr = config.learning_rate
    chosen_hidden = config.hidden_size
    chosen_dropout = config.dropout
    chosen_crf_reg = config.crf_regularization

    final_model: Optional[ADLMBiLSTMCRF] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    training_status = "completed"
    last_completed_epoch = 0

    try:
        if config.use_dragonfly:
            def objective(candidate: HyperCandidate) -> float:
                model = build_model(candidate)
                optimizer = torch.optim.Adam(model.parameters(), lr=candidate.learning_rate)

                # Quick inner-loop training during hyperparameter search.
                for _ in range(config.dragonfly_inner_epochs):
                    train_one_epoch(model, train_loader, optimizer, device)

                metrics = evaluate(model, val_loader, device)
                return metrics["f1"]

            optimizer_do = DragonflyHyperOptimizer(
                population=config.dragonfly_population,
                iterations=config.dragonfly_iterations,
                seed=config.seed,
            )
            best_candidate = optimizer_do.optimize(objective)
            chosen_lr = best_candidate.learning_rate
            chosen_hidden = best_candidate.hidden_size
            chosen_dropout = best_candidate.dropout
            chosen_crf_reg = best_candidate.crf_regularization
            LOGGER.info(
                "[Dragonfly] best candidate: %s",
                {
                    "learning_rate": chosen_lr,
                    "hidden_size": chosen_hidden,
                    "dropout": chosen_dropout,
                    "crf_regularization": chosen_crf_reg,
                },
            )

        final_model = ADLMBiLSTMCRF(
            vocab_size=vocabulary.size,
            embedding_dim=config.embedding_dim,
            hidden_size=chosen_hidden,
            dropout=chosen_dropout,
            temporal_dim=train_dataset.temporal_dim,
            categorical_dim=train_dataset.categorical_dim,
            contextual_dim=train_dataset.contextual_dim,
            num_labels=len(REPORT_LABELS),
            crf_regularization=chosen_crf_reg,
        ).to(device)

        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=chosen_lr)

        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(final_model, train_loader, final_optimizer, device)
            val_metrics = evaluate(final_model, val_loader, device)
            last_completed_epoch = epoch

            LOGGER.info(
                "Epoch %02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | "
                "val_precision=%.4f | val_recall=%.4f | val_f1=%.4f | val_kappa=%.4f",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["kappa"],
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_metrics = val_metrics
                best_state = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}

    except KeyboardInterrupt:
        training_status = "interrupted"
        LOGGER.warning("KeyboardInterrupt received. Saving checkpoint from current training state...")

        # If interruption happened very early (e.g., during hyperparameter search),
        # still initialize a model so a valid checkpoint is always produced.
        if final_model is None:
            final_model = ADLMBiLSTMCRF(
                vocab_size=vocabulary.size,
                embedding_dim=config.embedding_dim,
                hidden_size=chosen_hidden,
                dropout=chosen_dropout,
                temporal_dim=train_dataset.temporal_dim,
                categorical_dim=train_dataset.categorical_dim,
                contextual_dim=train_dataset.contextual_dim,
                num_labels=len(REPORT_LABELS),
                crf_regularization=chosen_crf_reg,
            ).to(device)

    if final_model is None:
        raise RuntimeError("Unable to initialize model for checkpoint saving.")

    latest_state = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
    if best_state is None:
        best_state = latest_state
        if not best_metrics:
            best_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "kappa": 0.0,
                "loss": 0.0,
            }

    state_to_save = latest_state if training_status == "interrupted" else best_state

    config.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_to_save,
            "best_model_state_dict": best_state,
            "vocabulary": vocabulary.token_to_id,
            "model_config": {
                "embedding_dim": config.embedding_dim,
                "hidden_size": chosen_hidden,
                "dropout": chosen_dropout,
                "learning_rate": chosen_lr,
                "crf_regularization": chosen_crf_reg,
                "max_seq_len": config.max_seq_len,
                "temporal_dim": train_dataset.temporal_dim,
                "categorical_dim": train_dataset.categorical_dim,
                "contextual_dim": train_dataset.contextual_dim,
                "label_names": REPORT_LABELS,
            },
            "best_metrics": best_metrics,
            "label_column": label_column,
            "label_mode": "binary_duplicate_vs_non_duplicate",
            "training_status": training_status,
            "last_completed_epoch": last_completed_epoch,
            "epochs_requested": config.epochs,
        },
        config.model_out,
    )

    if training_status == "interrupted":
        LOGGER.info("Saved interrupted checkpoint to: %s", config.model_out)
    else:
        LOGGER.info("Saved best model to: %s", config.model_out)
    LOGGER.info("Best validation metrics: %s", best_metrics)


def default_model_output_path() -> Path:
    """Build a unique checkpoint path using current date and time."""
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    return Path("models") / f"adlm_bilstm_crf_{timestamp}.pt"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="ADLM-style bug duplicate detection model (BiLSTM + CRF + optional Dragonfly tuning)."
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--data_paths",
        type=Path,
        nargs="+",
        help="One or more dataset CSV paths to use for this run.",
    )
    dataset_group.add_argument(
        "--data_path",
        type=Path,
        help="Single dataset CSV path to use for this run.",
    )
    parser.add_argument(
        "--prepared_data_out",
        type=Path,
        default=Path("dataset") / "prepared_binary_balanced.csv",
        help="Output CSV path for the cleaned + balanced binary dataset.",
    )
    parser.add_argument(
        "--start_training",
        action="store_true",
        help="Start training after preparation. If omitted, the script stops after data preparation.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional file path to save logs.",
    )
    parser.add_argument(
        "--model_out",
        type=Path,
        default=None,
        help="Optional checkpoint path. If omitted, a timestamped filename is generated automatically.",
    )
    parser.add_argument("--label_column", type=str, default=None)

    parser.add_argument("--max_vocab_size", type=int, default=15000)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--crf_regularization", type=float, default=1e-4)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_dragonfly", action="store_true")
    parser.add_argument("--dragonfly_population", type=int, default=5)
    parser.add_argument("--dragonfly_iterations", type=int, default=2)
    parser.add_argument("--dragonfly_inner_epochs", type=int, default=2)

    args = parser.parse_args()
    resolved_data_paths = resolve_training_data_paths(args.data_paths, args.data_path)
    resolved_model_out = args.model_out if args.model_out is not None else default_model_output_path()
    return TrainConfig(
        data_paths=resolved_data_paths,
        prepared_data_out=args.prepared_data_out,
        start_training=args.start_training,
        log_level=args.log_level,
        log_file=args.log_file,
        model_out=resolved_model_out,
        label_column=args.label_column,
        max_vocab_size=args.max_vocab_size,
        max_seq_len=args.max_seq_len,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        crf_regularization=args.crf_regularization,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_size=args.val_size,
        seed=args.seed,
        use_dragonfly=args.use_dragonfly,
        dragonfly_population=args.dragonfly_population,
        dragonfly_iterations=args.dragonfly_iterations,
        dragonfly_inner_epochs=args.dragonfly_inner_epochs,
    )


if __name__ == "__main__":
    train_config = parse_args()
    run_training(train_config)


