import json
import math
import os
import random
import re
import sys
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_text(text: str) -> List[str]:
    """Tokenizer used by the ADLM text branch with basic text normalization."""
    normalized = re.sub(r"[^\w\s]", " ", str(text).lower())
    tokens = re.findall(r"[a-z0-9_]+", normalized)
    return [token for token in tokens if token not in STOP_WORDS]


def preprocess_text_for_dataset(text: object) -> str:
    """Lowercase text, remove punctuation, and drop English stopwords."""
    return " ".join(tokenize_text(str(text)))


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


def infer_weak_duplicate_label(text: str) -> int:
    """
    Build weak supervision labels if the dataset has no explicit target column.
    1 => duplicate report, 0 => non-duplicate.
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
    return int(any(re.search(pattern, lower) for pattern in patterns))


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
    for fmt in ["%d/%b/%Y %I:%M %p", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None


def build_temporal_vector(row: pd.Series) -> List[float]:
    """
    Convert time values into a numeric vector.
    Uses cyclic encoding for calendar/time features to preserve periodicity.
    """
    payload = safe_json_load(row.get("temporal_features", ""))
    created_raw = str(payload.get("created_raw", "")) if payload else ""
    if not created_raw and "Created" in row:
        created_raw = str(row.get("Created", ""))

    dt = parse_datetime(created_raw)
    if dt is None:
        return [0.0] * 9

    year_scaled = (dt.year - 1990) / 50.0
    month_angle = 2.0 * math.pi * dt.month / 12.0
    weekday_angle = 2.0 * math.pi * dt.weekday() / 7.0
    hour_angle = 2.0 * math.pi * dt.hour / 24.0
    return [
        year_scaled,
        math.sin(month_angle),
        math.cos(month_angle),
        math.sin(weekday_angle),
        math.cos(weekday_angle),
        math.sin(hour_angle),
        math.cos(hour_angle),
        dt.minute / 59.0,
        float(dt.weekday() >= 5),
    ]


def build_categorical_vector(row: pd.Series, fallback_text: str) -> List[float]:
    """Build a fixed-width categorical vector from JSON features or fallback heuristics."""
    payload = safe_json_load(row.get("categorical_features", ""))
    vector_obj = payload.get("vector") if payload else None
    if isinstance(vector_obj, list):
        return [float(value) for value in vector_obj]

    priority = str(payload.get("priority", "")) if payload else ""
    severity = str(payload.get("severity", "")) if payload else ""
    issue_type = str(payload.get("issue_type", "")) if payload else ""
    component = str(payload.get("component", "")) if payload else ""

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
    Extract context cues from logs/code snippets/error traces.
    This keeps the contextual branch compact and robust.
    """
    contextual_text = str(row.get("contextual_features", "")).strip()
    if not contextual_text:
        contextual_text = str(row.get("Description", "")).strip()
    lower = contextual_text.lower()

    has_stack_trace = bool(re.search(r"\b(stack trace|traceback|call stack)\b", lower))
    has_error = bool(re.search(r"\b(error|exception|fatal|assert|segmentation fault|sigsegv)\b", lower))
    has_code = bool(
        re.search(r"\b0x[0-9a-f]{5,}\b", lower)
        or re.search(r"\b[A-Za-z_][\w\-.]*\.(c|cc|cpp|h|hpp|java|js|py|cs|go)\b", contextual_text)
        or re.search(r"[{}();]", contextual_text)
    )
    context_token_count = len(tokenize_text(contextual_text))
    normalized_length = min(context_token_count / 300.0, 1.0)

    return [
        float(has_stack_trace),
        float(has_error),
        float(has_code),
        normalized_length,
    ]


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

    # If the file is the converted corpus output, these columns already exist.
    if "textual_features" not in df.columns:
        summary = df.get("Summary", pd.Series([""] * len(df))).fillna("")
        description = df.get("Description", pd.Series([""] * len(df))).fillna("")
        df["textual_features"] = (summary.astype(str) + " " + description.astype(str)).str.strip()

    if "contextual_features" not in df.columns:
        df["contextual_features"] = df.get("Description", pd.Series([""] * len(df))).fillna("")

    return df


def prepare_eclipse_binary_dataset(
    input_path: Path,
    output_path: Path,
    seed: int,
    negative_to_positive_ratio: float = 1.0,
) -> Path:
    """
    Build a balanced Eclipse duplicate/non-duplicate CSV with cleaned text.
    """
    dataframe = pd.read_csv(input_path).copy()
    if "Duplicated_issue" not in dataframe.columns:
        raise ValueError("Expected 'Duplicated_issue' column in Eclipse dataset.")

    title = dataframe.get("Title", pd.Series([""] * len(dataframe))).fillna("")
    description = dataframe.get("Description", pd.Series([""] * len(dataframe))).fillna("")

    dataframe["Title"] = title.map(preprocess_text_for_dataset)
    dataframe["Description"] = description.map(preprocess_text_for_dataset)
    dataframe["textual_features"] = (dataframe["Title"] + " " + dataframe["Description"]).str.strip()
    dataframe["contextual_features"] = dataframe["Description"]
    dataframe["is_duplicate"] = (~dataframe["Duplicated_issue"].isna()).astype(int)

    dataframe = dataframe[dataframe["textual_features"].str.len() > 0].copy()

    positive = dataframe[dataframe["is_duplicate"] == 1]
    negative = dataframe[dataframe["is_duplicate"] == 0]
    if positive.empty or negative.empty:
        raise ValueError("Cannot balance classes because one class is empty.")

    ratio = max(negative_to_positive_ratio, 0.05)
    target_negative = min(len(negative), max(1, int(round(len(positive) * ratio))))
    sampled_negative = negative.sample(n=target_negative, random_state=seed, replace=False)

    balanced = pd.concat([positive, sampled_negative], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_path, index=False)

    print(f"Prepared dataset saved to: {output_path}")
    print("Prepared label counts:", balanced["is_duplicate"].value_counts().to_dict())
    return output_path


def detect_label_column(df: pd.DataFrame, explicit_label_column: Optional[str]) -> Optional[str]:
    """Resolve the supervision column if it exists in the dataset."""
    if explicit_label_column and explicit_label_column in df.columns:
        return explicit_label_column
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def normalize_label(value: object) -> Optional[int]:
    """Normalize labels from numeric/string columns into 0/1."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    if isinstance(value, (int, float)):
        return int(float(value) > 0)

    text = str(value).strip().lower()
    truthy = {"1", "true", "yes", "duplicate", "dup", "y"}
    falsy = {"0", "false", "no", "non-duplicate", "not duplicate", "n"}
    if text in truthy:
        return 1
    if text in falsy:
        return 0

    # Last fallback for labels like "Severity 3 - Minor" etc.
    if text.isdigit():
        return int(int(text) > 0)
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

        contextual_text = str(row.get("contextual_features", ""))
        label: Optional[int] = None
        if label_column:
            label = normalize_label(row.get(label_column))

        if label is None:
            # Weak supervision fallback keeps the training script runnable on unlabeled corpora.
            label = infer_weak_duplicate_label(text + " " + contextual_text)

        examples.append(
            BugExample(
                text=text,
                temporal_vector=build_temporal_vector(row),
                categorical_vector=build_categorical_vector(row, text + " " + contextual_text),
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
        num_labels: int = 2,
        crf_regularization: float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.crf_regularization = crf_regularization

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.text_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        side_input_dim = temporal_dim + categorical_dim + contextual_dim
        # Side-channel MLP aligns feature scales before fusion with token states.
        self.side_encoder = nn.Sequential(
            nn.Linear(max(1, side_input_dim), hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
        )

        # Fusion head maps [text_state || side_state] -> token emissions for CRF.
        self.emission_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_labels),
        )

        self.crf = CRF(num_labels, pad_idx=None, use_gpu=torch.cuda.is_available())

    def _crf_l2_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for parameter in self.crf.parameters():
            penalty = penalty + torch.sum(parameter ** 2)
        return penalty

    @staticmethod
    def token_tags_to_report_labels(tag_sequences: List[List[int]]) -> List[int]:
        """Collapse token predictions into one report-level duplicate decision."""
        predictions: List[int] = []
        for tags in tag_sequences:
            if not tags:
                predictions.append(0)
                continue
            positive_ratio = float(sum(tags)) / float(len(tags))
            predictions.append(int(positive_ratio >= 0.5))
        return predictions

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

        output: Dict[str, object] = {
            "decoded_tags": decoded_tags,
            "emissions": emissions,
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
    data_path: Path
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
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    use_amp: bool
    gradient_accumulation_steps: int
    use_weighted_sampler: bool
    prepare_eclipse_binary: bool
    prepared_data_out: Optional[Path]
    negative_to_positive_ratio: float


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
        average="binary",
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


def create_grad_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    """Create a GradScaler compatible with newer and older PyTorch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return cast(torch.cuda.amp.GradScaler, torch.amp.GradScaler("cuda", enabled=enabled))
        except TypeError:
            return cast(torch.cuda.amp.GradScaler, torch.amp.GradScaler(enabled=enabled))
    return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = False,
) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking)
    return moved


def train_one_epoch(
    model: ADLMBiLSTMCRF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    gradient_accumulation_steps: int,
    non_blocking_transfer: bool,
) -> float:
    model.train()
    losses: List[float] = []
    accumulation_steps = max(1, gradient_accumulation_steps)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device, non_blocking=non_blocking_transfer)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                temporal_features=batch["temporal_features"],
                categorical_features=batch["categorical_features"],
                contextual_features=batch["contextual_features"],
                tags=batch["tag_ids"],
            )
            loss = cast(torch.Tensor, output["loss"]) / float(accumulation_steps)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step % accumulation_steps == 0) or (step == len(dataloader))
        if should_step:
            if scaler is not None and use_amp and device.type == "cuda":
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu().item()) * float(accumulation_steps))

    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: ADLMBiLSTMCRF,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    non_blocking_transfer: bool,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    targets: List[int] = []
    predictions: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device, non_blocking=non_blocking_transfer)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    temporal_features=batch["temporal_features"],
                    categorical_features=batch["categorical_features"],
                    contextual_features=batch["contextual_features"],
                    tags=batch["tag_ids"],
                )

            if "loss" in output:
                losses.append(float(cast(torch.Tensor, output["loss"]).detach().cpu().item()))

            decoded = output["decoded_tags"]
            batch_predictions = ADLMBiLSTMCRF.token_tags_to_report_labels(decoded)
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
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    use_weighted_sampler: bool,
) -> DataArtifacts:
    vocabulary = Vocabulary()
    vocabulary.build([example.text for example in train_examples], max_size=max_vocab_size)

    train_dataset = BugReportDataset(train_examples, vocabulary=vocabulary, max_seq_len=max_seq_len)
    val_dataset = BugReportDataset(val_examples, vocabulary=vocabulary, max_seq_len=max_seq_len)

    resolved_workers = max(0, num_workers)
    loader_kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": resolved_workers,
        "pin_memory": pin_memory,
    }
    if resolved_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(2, prefetch_factor)

    if use_weighted_sampler:
        labels = np.array([example.label for example in train_examples], dtype=np.int64)
        class_counts = np.bincount(labels, minlength=2).astype(np.float64)
        class_counts[class_counts == 0.0] = 1.0
        sample_weights = np.array([1.0 / class_counts[label] for label in labels], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return DataArtifacts(
        vocabulary=vocabulary,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)

    effective_data_path = config.data_path
    if config.prepare_eclipse_binary:
        prepared_path = config.prepared_data_out
        if prepared_path is None:
            prepared_path = config.data_path.with_name(config.data_path.stem + "_binary_balanced_preprocessed.csv")
        effective_data_path = prepare_eclipse_binary_dataset(
            input_path=config.data_path,
            output_path=prepared_path,
            seed=config.seed,
            negative_to_positive_ratio=config.negative_to_positive_ratio,
        )

    dataframe = load_modeling_dataframe(effective_data_path)
    label_column = detect_label_column(dataframe, config.label_column)
    examples = build_examples(dataframe, label_column=label_column)
    if len(examples) < 10:
        raise ValueError("Not enough valid training examples found in the dataset.")

    train_examples, val_examples = split_examples(examples, val_size=config.val_size, seed=config.seed)
    print("Train label counts:", Counter([example.label for example in train_examples]))
    print("Validation label counts:", Counter([example.label for example in val_examples]))

    effective_num_workers = config.num_workers
    effective_persistent_workers = config.persistent_workers
    if os.name == "nt" and sys.argv and sys.argv[0] == "-c" and effective_num_workers > 0:
        print("Detected Windows -c execution; forcing num_workers=0 to avoid DataLoader worker hangs.")
        effective_num_workers = 0
        effective_persistent_workers = False

    artifacts = build_dataloaders(
        train_examples=train_examples,
        val_examples=val_examples,
        max_vocab_size=config.max_vocab_size,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        num_workers=effective_num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=effective_persistent_workers,
        prefetch_factor=config.prefetch_factor,
        use_weighted_sampler=config.use_weighted_sampler,
    )

    train_dataset = artifacts.train_dataset
    train_loader = artifacts.train_loader
    val_loader = artifacts.val_loader
    vocabulary = artifacts.vocabulary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.use_amp and device.type == "cuda")
    non_blocking_transfer = bool(config.pin_memory and device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

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
                candidate_scaler = create_grad_scaler(use_amp)

                # Quick inner-loop training during hyperparameter search.
                for _ in range(config.dragonfly_inner_epochs):
                    train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        device,
                        scaler=candidate_scaler,
                        use_amp=use_amp,
                        gradient_accumulation_steps=1,
                        non_blocking_transfer=non_blocking_transfer,
                    )

                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    use_amp=use_amp,
                    non_blocking_transfer=non_blocking_transfer,
                )
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
            print(
                "[Dragonfly] best candidate:",
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
            crf_regularization=chosen_crf_reg,
        ).to(device)

        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=chosen_lr)
        final_scaler = create_grad_scaler(use_amp)

        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(
                final_model,
                train_loader,
                final_optimizer,
                device,
                scaler=final_scaler,
                use_amp=use_amp,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                non_blocking_transfer=non_blocking_transfer,
            )
            val_metrics = evaluate(
                final_model,
                val_loader,
                device,
                use_amp=use_amp,
                non_blocking_transfer=non_blocking_transfer,
            )
            last_completed_epoch = epoch

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_precision={val_metrics['precision']:.4f} | "
                f"val_recall={val_metrics['recall']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_kappa={val_metrics['kappa']:.4f}"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_metrics = val_metrics
                best_state = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}

    except KeyboardInterrupt:
        training_status = "interrupted"
        print("\nKeyboardInterrupt received. Saving checkpoint from current training state...")

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
            },
            "best_metrics": best_metrics,
            "label_column": label_column,
            "label_mode": "explicit" if label_column else "weak_supervision",
            "training_status": training_status,
            "last_completed_epoch": last_completed_epoch,
            "epochs_requested": config.epochs,
        },
        config.model_out,
    )

    if training_status == "interrupted":
        print("Saved interrupted checkpoint to:", config.model_out)
    else:
        print("Saved best model to:", config.model_out)
    print("Best validation metrics:", best_metrics)


def _default_data_path() -> Path:
    """Resolve a sensible default dataset path for local runs and Colab."""
    candidates = [
        Path("/content/dataset/corpus_features.csv"),
        Path("dataset") / "corpus_features.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("dataset") / "corpus_features.csv"


def _default_model_out() -> Path:
    """Resolve a sensible default checkpoint path for local runs and Colab."""
    if Path("/content").exists():
        return Path("/content/models/adlm_bilstm_crf.pt")
    return Path("models") / "adlm_bilstm_crf.pt"


def _as_int(value: object) -> int:
    return int(cast(int | float | str, value))


def _as_float(value: object) -> float:
    return float(cast(int | float | str, value))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _as_optional_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _as_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def build_colab_train_config(config_overrides: Optional[Dict[str, object]] = None) -> TrainConfig:
    """
    Build TrainConfig without argparse so the script can run directly in Colab.
    Pass a dictionary of overrides for any field in TrainConfig.
    """
    config_values: Dict[str, object] = {
        "data_path": _default_data_path(),
        "model_out": _default_model_out(),
        "label_column": "label",
        "max_vocab_size": 15000,
        "max_seq_len": 128,
        "embedding_dim": 128,
        "hidden_size": 128,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "crf_regularization": 1e-4,
        "batch_size": 16,
        "epochs": 8,
        "val_size": 0.2,
        "seed": 42,
        "use_dragonfly": False,
        "dragonfly_population": 5,
        "dragonfly_iterations": 2,
        "dragonfly_inner_epochs": 2,
        "num_workers": max(2, min(8, (os.cpu_count() or 2) // 2)),
        "prefetch_factor": 4,
        "persistent_workers": True,
        "pin_memory": True,
        "use_amp": True,
        "gradient_accumulation_steps": 1,
        "use_weighted_sampler": True,
        "prepare_eclipse_binary": False,
        "prepared_data_out": None,
        "negative_to_positive_ratio": 1.0,
    }

    if config_overrides:
        config_values.update(config_overrides)

    return TrainConfig(
        data_path=_as_path(config_values["data_path"]),
        model_out=_as_path(config_values["model_out"]),
        label_column=_as_optional_str(config_values["label_column"]),
        max_vocab_size=_as_int(config_values["max_vocab_size"]),
        max_seq_len=_as_int(config_values["max_seq_len"]),
        embedding_dim=_as_int(config_values["embedding_dim"]),
        hidden_size=_as_int(config_values["hidden_size"]),
        dropout=_as_float(config_values["dropout"]),
        learning_rate=_as_float(config_values["learning_rate"]),
        crf_regularization=_as_float(config_values["crf_regularization"]),
        batch_size=_as_int(config_values["batch_size"]),
        epochs=_as_int(config_values["epochs"]),
        val_size=_as_float(config_values["val_size"]),
        seed=_as_int(config_values["seed"]),
        use_dragonfly=_as_bool(config_values["use_dragonfly"]),
        dragonfly_population=_as_int(config_values["dragonfly_population"]),
        dragonfly_iterations=_as_int(config_values["dragonfly_iterations"]),
        dragonfly_inner_epochs=_as_int(config_values["dragonfly_inner_epochs"]),
        num_workers=max(0, _as_int(config_values["num_workers"])),
        prefetch_factor=max(2, _as_int(config_values["prefetch_factor"])),
        persistent_workers=_as_bool(config_values["persistent_workers"]),
        pin_memory=_as_bool(config_values["pin_memory"]),
        use_amp=_as_bool(config_values["use_amp"]),
        gradient_accumulation_steps=max(1, _as_int(config_values["gradient_accumulation_steps"])),
        use_weighted_sampler=_as_bool(config_values["use_weighted_sampler"]),
        prepare_eclipse_binary=_as_bool(config_values["prepare_eclipse_binary"]),
        prepared_data_out=_as_optional_path(config_values["prepared_data_out"]),
        negative_to_positive_ratio=max(0.05, _as_float(config_values["negative_to_positive_ratio"])),
    )


def run_colab(config_overrides: Optional[Dict[str, object]] = None) -> None:
    """Run training directly with inline configuration (Colab-friendly)."""
    config = build_colab_train_config(config_overrides)
    print("Running with config:", config)
    run_training(config)


if __name__ == "__main__":
    # Edit this dictionary in Colab before running this file.
    # Example:
    # COLAB_CONFIG_OVERRIDES = {
    #     "data_path": "/content/drive/MyDrive/Bug Report AI model/dataset/corpus_features.csv",
    #     "model_out": "/content/drive/MyDrive/Bug Report AI model/models/adlm_colab.pt",
    #     "epochs": 20,
    #     "batch_size": 64,
    # }
    COLAB_CONFIG_OVERRIDES: Dict[str, object] = {}
    run_colab(COLAB_CONFIG_OVERRIDES)


