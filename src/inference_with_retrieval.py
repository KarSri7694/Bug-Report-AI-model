import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model import (
    ADLMBiLSTMCRF,
    Vocabulary,
    build_categorical_vector,
    build_contextual_vector,
    build_temporal_vector,
    load_modeling_dataframe,
)


def restore_vocabulary(token_to_id: Dict[str, int]) -> Vocabulary:
    """Rebuild Vocabulary object from checkpoint token-id mapping."""
    if not token_to_id:
        raise ValueError("Checkpoint vocabulary is empty.")

    max_id = max(int(idx) for idx in token_to_id.values())
    id_to_token: List[str] = [Vocabulary.UNK] * (max_id + 1)
    for token, idx in token_to_id.items():
        id_to_token[int(idx)] = token

    vocab = Vocabulary()
    vocab.token_to_id = {str(token): int(idx) for token, idx in token_to_id.items()}
    vocab.id_to_token = id_to_token
    return vocab


def load_model_for_inference(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[ADLMBiLSTMCRF, Vocabulary, Dict[str, Any]]:
    """Load trained model + vocabulary + config from saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_config" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'model_config'.")
    if "vocabulary" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'vocabulary'.")

    model_config = checkpoint["model_config"]
    vocabulary = restore_vocabulary(checkpoint["vocabulary"])

    model = ADLMBiLSTMCRF(
        vocab_size=vocabulary.size,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_size=int(model_config["hidden_size"]),
        dropout=float(model_config["dropout"]),
        temporal_dim=int(model_config["temporal_dim"]),
        categorical_dim=int(model_config["categorical_dim"]),
        contextual_dim=int(model_config["contextual_dim"]),
        crf_regularization=float(model_config.get("crf_regularization", 0.0)),
    ).to(device)

    state_dict = checkpoint.get("best_model_state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Invalid checkpoint: missing model state dictionary.")

    model.load_state_dict(state_dict)
    model.eval()
    return model, vocabulary, model_config


def build_manual_row(
    text: str,
    contextual_text: str,
    created_raw: str,
    priority: str,
    severity: str,
    issue_type: str,
    component: str,
) -> pd.Series:
    """Construct a synthetic row so existing feature builders can be reused."""
    temporal_payload = json.dumps({"created_raw": created_raw}) if created_raw else ""

    categorical_payload = ""
    if priority or severity or issue_type or component:
        categorical_payload = json.dumps(
            {
                "priority": priority,
                "severity": severity,
                "issue_type": issue_type,
                "component": component,
            }
        )

    return pd.Series(
        {
            "textual_features": text,
            "temporal_features": temporal_payload,
            "categorical_features": categorical_payload,
            "contextual_features": contextual_text,
            "Description": contextual_text,
            "Priority": priority,
            "Severity": severity,
            "Issue Type": issue_type,
            "Component/s": component,
        }
    )


def predict_duplicate(
    model: ADLMBiLSTMCRF,
    vocabulary: Vocabulary,
    model_config: Dict[str, Any],
    row: pd.Series,
    text: str,
    contextual_text: str,
    classifier_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Run model inference and collapse token tags to one report-level prediction."""
    max_seq_len = int(model_config["max_seq_len"])
    encoded = vocabulary.encode(text, max_seq_len=max_seq_len)

    temporal_vector = build_temporal_vector(row)
    categorical_vector = build_categorical_vector(row, text + " " + contextual_text)
    contextual_vector = build_contextual_vector(row)

    batch = {
        "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
        "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
        "temporal_features": torch.tensor([temporal_vector], dtype=torch.float32, device=device),
        "categorical_features": torch.tensor([categorical_vector], dtype=torch.float32, device=device),
        "contextual_features": torch.tensor([contextual_vector], dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            temporal_features=batch["temporal_features"],
            categorical_features=batch["categorical_features"],
            contextual_features=batch["contextual_features"],
        )

    decoded_tags = output["decoded_tags"][0] if output["decoded_tags"] else []
    token_count = len(decoded_tags)
    positive_ratio = (float(sum(decoded_tags)) / float(token_count)) if token_count > 0 else 0.0
    classifier_prediction = int(positive_ratio >= classifier_threshold)

    return {
        "classifier_prediction": classifier_prediction,
        "classifier_positive_ratio": positive_ratio,
        "classifier_threshold": classifier_threshold,
        "decoded_token_count": token_count,
    }


def retrieve_option_2(
    database_df: pd.DataFrame,
    query_text: str,
    query_context: str,
    top_k: int,
    skip_report_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieval option 2:
    TF-IDF vectorization on textual + contextual content and cosine similarity search.
    """
    text_col = database_df.get("textual_features", pd.Series([""] * len(database_df))).fillna("").astype(str)
    context_col = database_df.get("contextual_features", pd.Series([""] * len(database_df))).fillna("").astype(str)
    corpus = (text_col + " " + context_col).str.strip().tolist()
    if not corpus:
        return []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(corpus)

    query_vector = vectorizer.transform([(query_text + " " + query_context).strip()])
    similarities = cosine_similarity(query_vector, matrix).ravel()

    if skip_report_id is not None and "report_id" in database_df.columns:
        matches = database_df.index[database_df["report_id"].astype(str) == str(skip_report_id)].tolist()
        if matches:
            similarities[matches[0]] = -1.0

    top_k = max(1, min(top_k, len(similarities)))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < 0:
            continue
        row = database_df.iloc[int(idx)]
        raw_report_id = row.get("report_id", idx)
        try:
            report_id = int(raw_report_id)
        except (TypeError, ValueError):
            report_id = int(idx)

        raw_label = row.get("label", None)
        label_value: Optional[int] = None
        if raw_label is not None and not pd.isna(raw_label):
            try:
                label_value = int(float(raw_label))
            except (TypeError, ValueError):
                label_value = None

        snippet = str(row.get("textual_features", "")).replace("\n", " ").strip()[:220]
        results.append(
            {
                "rank": len(results) + 1,
                "report_id": report_id,
                "similarity": round(score, 6),
                "label": label_value,
                "snippet": snippet,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ADLM model inference for one bug report and optionally enable retrieval option 2 "
            "(TF-IDF + cosine similarity over the bug database)."
        )
    )

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--database", type=Path, default=Path("dataset") / "corpus_features.csv")

    parser.add_argument("--bug_text", type=str, default="", help="Text of the incoming bug report")
    parser.add_argument("--bug_text_file", type=Path, default=None, help="Optional text file for bug text")
    parser.add_argument(
        "--query_report_id",
        type=int,
        default=None,
        help="Use an existing report_id from --database as query (useful for quick testing)",
    )

    parser.add_argument("--contextual_text", type=str, default="")
    parser.add_argument("--created_raw", type=str, default="")
    parser.add_argument("--priority", type=str, default="")
    parser.add_argument("--severity", type=str, default="")
    parser.add_argument("--issue_type", type=str, default="")
    parser.add_argument("--component", type=str, default="")

    parser.add_argument("--retrieval_option", type=int, choices=[0, 2], default=2)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--classifier_threshold", type=float, default=0.5)
    parser.add_argument("--similarity_threshold", type=float, default=0.35)

    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def resolve_query(args: argparse.Namespace, database_df: pd.DataFrame) -> Tuple[pd.Series, str, str, Optional[int]]:
    """Resolve query row/text either from report_id in DB or from manual input."""
    if args.query_report_id is not None:
        if "report_id" not in database_df.columns:
            raise ValueError("--query_report_id was provided but database has no 'report_id' column.")

        matched = database_df[database_df["report_id"].astype(str) == str(args.query_report_id)]
        if matched.empty:
            raise ValueError(f"report_id={args.query_report_id} not found in database.")

        row = matched.iloc[0]
        text = str(row.get("textual_features", "")).strip()
        context = str(row.get("contextual_features", "")).strip()
        if not text:
            raise ValueError("Selected report has empty textual_features.")
        return row, text, context, args.query_report_id

    bug_text = str(args.bug_text or "").strip()
    if args.bug_text_file is not None:
        bug_text = args.bug_text_file.read_text(encoding="utf-8").strip()

    if not bug_text:
        raise ValueError("Provide --bug_text, --bug_text_file, or --query_report_id.")

    row = build_manual_row(
        text=bug_text,
        contextual_text=str(args.contextual_text or "").strip(),
        created_raw=str(args.created_raw or "").strip(),
        priority=str(args.priority or "").strip(),
        severity=str(args.severity or "").strip(),
        issue_type=str(args.issue_type or "").strip(),
        component=str(args.component or "").strip(),
    )
    return row, bug_text, str(args.contextual_text or "").strip(), None


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    database_df = load_modeling_dataframe(args.database)

    model, vocabulary, model_config = load_model_for_inference(args.checkpoint, device)
    query_row, query_text, query_context, query_report_id = resolve_query(args, database_df)

    classification = predict_duplicate(
        model=model,
        vocabulary=vocabulary,
        model_config=model_config,
        row=query_row,
        text=query_text,
        contextual_text=query_context,
        classifier_threshold=args.classifier_threshold,
        device=device,
    )

    retrieval_results: List[Dict[str, Any]] = []
    retrieval_duplicate = 0
    top_similarity = 0.0

    if args.retrieval_option == 2:
        retrieval_results = retrieve_option_2(
            database_df=database_df,
            query_text=query_text,
            query_context=query_context,
            top_k=args.top_k,
            skip_report_id=query_report_id,
        )
        if retrieval_results:
            top_similarity = float(retrieval_results[0]["similarity"])
            retrieval_duplicate = int(top_similarity >= args.similarity_threshold)

    final_prediction = int(
        classification["classifier_prediction"] == 1 or retrieval_duplicate == 1
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "retrieval_option": args.retrieval_option,
        "final_prediction": final_prediction,
        "final_prediction_name": "duplicate" if final_prediction == 1 else "unique",
        "classifier": classification,
        "retrieval": {
            "enabled": args.retrieval_option == 2,
            "similarity_threshold": args.similarity_threshold,
            "retrieval_prediction": retrieval_duplicate,
            "top_similarity": top_similarity,
            "top_k": args.top_k,
            "matches": retrieval_results,
        },
    }

    output_text = json.dumps(result, indent=2)
    print(output_text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output_text + "\n", encoding="utf-8")
        print(f"Saved inference output to: {args.output_json}")


if __name__ == "__main__":
    main()