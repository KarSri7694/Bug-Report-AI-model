from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (
    ADLMBiLSTMCRF,
    BugReportDataset,
    COMPONENT_LABELS,
    PRIORITY_LABELS,
    REPORT_LABELS,
    SEVERITY_LABELS,
    TYPE_LABELS,
    build_dataloaders,
    build_examples,
    detect_label_column,
    load_modeling_dataframe,
    one_hot,
    split_examples,
)


RAW_BUGZILLA_PATH = Path("dataset/bugzilla_dataset.csv")
BUGZILLA_4FEATURE_PATH = Path("dataset/bugzilla_dataset_4features.csv")
PREDICTIONS_PATH = Path("dataset/bugzilla_dataset_predictions.csv")

TRAINING_DATA_PATH = Path("dataset/Lucene_preprocessed_data_4features.csv")
CHECKPOINT_PATH = Path("models/adlm_bilstm_crf_17042026_130432.pt")


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_bugzilla_date(raw: str) -> datetime | None:
    text = safe_text(raw)
    if not text:
        return None

    for fmt in ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def normalize_priority(priority_raw: str) -> tuple[str, int]:
    text = safe_text(priority_raw).lower()
    if text.startswith("p") and text[1:].isdigit():
        pid = int(text[1:])
    elif text.isdigit():
        pid = int(text)
    else:
        pid = 0

    id_to_label = {1: "critical", 2: "high", 3: "medium", 4: "low", 5: "low"}
    return id_to_label.get(pid, "unknown"), pid


def normalize_issue_type(raw_type: str, text_hint: str) -> str:
    text = (safe_text(raw_type) + " " + safe_text(text_hint)).lower()
    if "regression" in text:
        return "regression"
    if "security" in text or "vulnerab" in text:
        return "security"
    if "performance" in text:
        return "performance"
    if "ui" in text:
        return "ui"
    if "feature" in text or "enhancement" in text:
        return "feature"
    if "build" in text:
        return "build"
    return "bug"


def normalize_component(component_raw: str) -> str:
    text = safe_text(component_raw).lower()
    if "ui" in text or "front" in text:
        return "ui"
    if "network" in text:
        return "networking"
    if "browser" in text:
        return "browser"
    if "layout" in text:
        return "layout"
    if "javascript" in text:
        return "javascript"
    if "qa" in text:
        return "qa"
    if text:
        return "general"
    return "unknown"


def build_first_seen_id_map(values: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = OrderedDict()
    next_id = 1
    for raw in values:
        norm = safe_text(raw).lower()
        if not norm:
            continue
        if norm not in mapping:
            mapping[norm] = next_id
            next_id += 1
    return mapping


def encode_context(raw: str, mapping: Dict[str, int]) -> int:
    key = safe_text(raw).lower()
    if not key:
        return 0
    return mapping.get(key, 0)


def preprocess_bugzilla_to_four_features() -> pd.DataFrame:
    if not RAW_BUGZILLA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {RAW_BUGZILLA_PATH}")

    df = pd.read_csv(RAW_BUGZILLA_PATH)
    df.columns = [str(column).strip() for column in df.columns]

    company_values = [safe_text(value) for value in df.get("company", pd.Series([""] * len(df))).tolist()]
    status_values = [safe_text(value) for value in df.get("status", pd.Series([""] * len(df))).tolist()]
    component_values = [safe_text(value) for value in df.get("component", pd.Series([""] * len(df))).tolist()]

    assigned_to_id_map = build_first_seen_id_map(company_values)
    fix_version_id_map = build_first_seen_id_map(status_values)
    components_id_map = build_first_seen_id_map(component_values)

    parsed_dates = [parse_bugzilla_date(raw) for raw in df.get("open_date", pd.Series([""] * len(df))).tolist()]
    anchor_date = next((value for value in parsed_dates if value is not None), None)
    epoch_start = datetime(1970, 1, 1)

    rows: List[Dict[str, object]] = []
    for index, row in df.iterrows():
        bug_id = safe_text(row.get("bug_id", ""))
        title = safe_text(row.get("title", ""))
        description = safe_text(row.get("description", ""))
        product = safe_text(row.get("product", ""))
        company = safe_text(row.get("company", ""))
        component = safe_text(row.get("component", ""))
        bug_type = safe_text(row.get("bug_type", ""))
        priority_raw = safe_text(row.get("priority", ""))
        status = safe_text(row.get("status", ""))
        open_date = safe_text(row.get("open_date", ""))
        is_duplicate = row.get("is_duplicate", 0)

        text_value = f"{title} {description}".strip()

        priority_label, priority_id = normalize_priority(priority_raw)
        issue_type_label = normalize_issue_type(bug_type, text_value)
        component_label = normalize_component(component)
        categorical_vector = (
            one_hot(priority_label, PRIORITY_LABELS)
            + one_hot("unknown", SEVERITY_LABELS)
            + one_hot(issue_type_label, TYPE_LABELS)
            + one_hot(component_label, COMPONENT_LABELS)
        )
        categorical_payload = {
            "priority_name": priority_raw,
            "priority_id": priority_id,
            "priority": priority_label,
            "issue_type": issue_type_label,
            "component": component_label,
            "vector": categorical_vector,
        }

        created_dt = parse_bugzilla_date(open_date)
        days_since_epoch = 0.0
        gap_from_anchor_hours = 0.0
        if created_dt is not None:
            days_since_epoch = float((created_dt - epoch_start).days)
            if anchor_date is not None:
                gap_from_anchor_hours = float((created_dt - anchor_date).total_seconds() / 3600.0)

        temporal_payload = {
            "created_raw": open_date,
            "bug_id": bug_id,
            "bug_id_numeric": int(float(bug_id)) if bug_id and bug_id.isdigit() else 0,
            "created_days_since_epoch": days_since_epoch,
            "gap_from_lucene1_hours": gap_from_anchor_hours,
        }

        contextual_payload = {
            "assigned_to": company,
            "assigned_to_id": encode_context(company, assigned_to_id_map),
            "fix_version": status,
            "fix_version_id": encode_context(status, fix_version_id_map),
            "components": component,
            "components_id": encode_context(component, components_id_map),
            "product": product,
        }

        rows.append(
            {
                "report_id": f"BUGZILLA-{bug_id or index + 1}",
                "textual_features": text_value,
                "categorical_features": json.dumps(categorical_payload, ensure_ascii=True),
                "temporal_features": json.dumps(temporal_payload, ensure_ascii=True),
                "contextual_features": json.dumps(contextual_payload, ensure_ascii=True),
                "is_duplicate": is_duplicate,
            }
        )

    output_df = pd.DataFrame(rows)
    BUGZILLA_4FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(BUGZILLA_4FEATURE_PATH, index=False)
    return output_df


def run_inference_on_bugzilla() -> pd.DataFrame:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training reference data not found: {TRAINING_DATA_PATH}")
    if not BUGZILLA_4FEATURE_PATH.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {BUGZILLA_4FEATURE_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    training_df = load_modeling_dataframe(TRAINING_DATA_PATH)
    train_label_column = detect_label_column(training_df, explicit_label_column=None)
    if train_label_column is None:
        raise ValueError("Could not detect training labels in Lucene preprocessed dataset.")

    training_examples = build_examples(training_df, train_label_column)
    train_examples, val_examples = split_examples(training_examples, val_size=0.2, seed=42)
    artifacts = build_dataloaders(
        train_examples=train_examples,
        val_examples=val_examples,
        max_vocab_size=15000,
        max_seq_len=128,
        batch_size=16,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model_config: Dict[str, float] = checkpoint["model_config"]
    model = ADLMBiLSTMCRF(
        vocab_size=artifacts.vocabulary.size,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_size=int(model_config["hidden_size"]),
        dropout=float(model_config["dropout"]),
        temporal_dim=int(model_config["temporal_dim"]),
        categorical_dim=int(model_config["categorical_dim"]),
        contextual_dim=int(model_config["contextual_dim"]),
        num_labels=len(REPORT_LABELS),
        crf_regularization=float(model_config.get("crf_regularization", 0.0)),
    )

    state = checkpoint.get("best_model_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    inference_df = load_modeling_dataframe(BUGZILLA_4FEATURE_PATH)
    inference_label_column = detect_label_column(inference_df, explicit_label_column=None)
    inference_examples = build_examples(inference_df, inference_label_column)

    inference_dataset = BugReportDataset(
        inference_examples,
        vocabulary=artifacts.vocabulary,
        max_seq_len=128,
    )
    inference_loader = DataLoader(inference_dataset, batch_size=16, shuffle=False)

    pred_ids: List[int] = []
    prob_rows: List[List[float]] = []
    with torch.no_grad():
        for batch in inference_loader:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                temporal_features=batch["temporal_features"],
                categorical_features=batch["categorical_features"],
                contextual_features=batch["contextual_features"],
                tags=batch["tag_ids"],
            )
            class_probs = output["class_probabilities"]
            if not isinstance(class_probs, torch.Tensor):
                raise TypeError("Expected class_probabilities tensor during inference.")

            prob_rows.extend(class_probs.detach().cpu().tolist())
            pred_ids.extend(class_probs.argmax(dim=1).detach().cpu().tolist())

    label_from_id = {idx: label for idx, label in enumerate(REPORT_LABELS)}

    output_df = pd.read_csv(BUGZILLA_4FEATURE_PATH).copy()
    output_df = output_df.reset_index(drop=True)
    output_df["predicted_label"] = [label_from_id.get(int(index), "unknown") for index in pred_ids]
    output_df["p_duplicate"] = [row[0] for row in prob_rows]
    output_df["p_bug"] = [row[1] for row in prob_rows]
    output_df["p_enhancement"] = [row[2] for row in prob_rows]

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(PREDICTIONS_PATH, index=False)
    return output_df


def main() -> None:
    feature_df = preprocess_bugzilla_to_four_features()
    prediction_df = run_inference_on_bugzilla()

    print(f"Saved preprocessed 4-feature dataset to: {BUGZILLA_4FEATURE_PATH}")
    print(f"Rows in preprocessed dataset: {len(feature_df)}")
    print(f"Saved prediction output to: {PREDICTIONS_PATH}")
    print(f"Rows scored: {len(prediction_df)}")

    summary = prediction_df["predicted_label"].value_counts(dropna=False).to_dict()
    print("Predicted label distribution:", summary)
    display_cols = [
        column
        for column in [
            "report_id",
            "is_duplicate",
            "predicted_label",
            "p_duplicate",
            "p_bug",
            "p_enhancement",
        ]
        if column in prediction_df.columns
    ]
    print("\nSample output rows:")
    print(prediction_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()