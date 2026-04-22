from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (
    ADLMBiLSTMCRF,
    BugReportDataset,
    REPORT_LABELS,
    build_dataloaders,
    build_examples,
    detect_label_column,
    infer_weak_report_label,
    load_modeling_dataframe,
    normalize_label,
    split_examples,
)


DATASET_PATH = Path("dataset/Lucene_preprocessed_data_4features.csv")
CHECKPOINT_PATH = Path("models/adlm_bilstm_crf_17042026_130432.pt")
SMALL_TEST_PATH = Path("dataset/test_small_lucene_4features.csv")
PREDICTIONS_PATH = Path("dataset/test_small_lucene_predictions.csv")


def row_label_id(row: pd.Series, label_column: Optional[str]) -> int:
    text = str(row.get("textual_features", "")).strip()
    contextual = str(row.get("contextual_text_plain", row.get("contextual_features", ""))).strip()
    merged = f"{text} {contextual}".strip()

    if label_column:
        normalized = normalize_label(row.get(label_column), merged)
        if normalized is not None:
            return int(normalized)
    return int(infer_weak_report_label(merged))


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    full_df = load_modeling_dataframe(DATASET_PATH)
    label_column = detect_label_column(full_df, explicit_label_column=None)
    if label_column is None:
        raise ValueError("No label column found. Expected a column like Resolution.")

    full_df = full_df.copy()
    full_df["__label_id"] = full_df.apply(lambda row: row_label_id(row, label_column), axis=1)

    selected_indices: List[int] = []
    for class_id in [0, 1, 2]:
        class_rows = full_df[full_df["__label_id"] == class_id]
        if class_rows.empty:
            continue
        sample_n = min(3, len(class_rows))
        sampled = class_rows.sample(n=sample_n, random_state=42)
        selected_indices.extend(sampled.index.tolist())

    selected_indices = sorted(selected_indices)
    small_df = full_df.loc[selected_indices].copy().reset_index(drop=True)

    keep_columns = [
        column
        for column in [
            "report_id",
            "textual_features",
            "categorical_features",
            "temporal_features",
            "contextual_features",
            label_column,
        ]
        if column in small_df.columns
    ]

    small_export = small_df[keep_columns].copy()
    SMALL_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    small_export.to_csv(SMALL_TEST_PATH, index=False)

    examples = build_examples(full_df, label_column)
    train_examples, val_examples = split_examples(examples, val_size=0.2, seed=42)
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

    test_df = load_modeling_dataframe(SMALL_TEST_PATH).reset_index(drop=True)
    test_examples = build_examples(test_df, label_column)
    test_dataset = BugReportDataset(test_examples, vocabulary=artifacts.vocabulary, max_seq_len=128)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    pred_ids: List[int] = []
    true_ids: List[int] = []
    probs_rows: List[List[float]] = []

    with torch.no_grad():
        for batch in test_loader:
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
                raise TypeError("Expected class_probabilities to be a tensor.")

            probs_rows.extend(class_probs.detach().cpu().tolist())
            pred_ids.extend(class_probs.argmax(dim=1).detach().cpu().tolist())
            true_ids.extend(batch["label"].detach().cpu().tolist())

    label_from_id = {index: name for index, name in enumerate(REPORT_LABELS)}
    output_df = small_export.copy().reset_index(drop=True)
    output_df["true_label"] = [label_from_id.get(int(i), "unknown") for i in true_ids]
    output_df["predicted_label"] = [label_from_id.get(int(i), "unknown") for i in pred_ids]
    output_df["p_duplicate"] = [row[0] for row in probs_rows]
    output_df["p_bug"] = [row[1] for row in probs_rows]
    output_df["p_enhancement"] = [row[2] for row in probs_rows]

    output_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved small test dataset to: {SMALL_TEST_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")
    print("\nPrediction output:")
    print(
        output_df[
            [
                column
                for column in [
                    "report_id",
                    label_column,
                    "true_label",
                    "predicted_label",
                    "p_duplicate",
                    "p_bug",
                    "p_enhancement",
                ]
                if column in output_df.columns
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()