# Model Documentation

- Source file: [src/model.py](src/model.py)
- Purpose: Train an ADLM-style bug report duplicate detector using text + temporal + categorical + contextual signals, with a BiLSTM-CRF model and optional Dragonfly-inspired hyperparameter tuning.

## 1. What This Module Does

[src/model.py](src/model.py) is a complete training script. It handles:

1. Loading and normalizing bug report data from CSV.
2. Building feature branches (text, temporal, categorical, contextual).
3. Converting examples into PyTorch datasets and dataloaders.
4. Defining the ADLM-inspired BiLSTM + CRF architecture.
5. Training and validation with report-level metrics.
6. Optional Dragonfly-style hyperparameter optimization.
7. Saving checkpoints with model state, vocabulary, config, and metrics.

## 2. High-Level Pipeline

The training flow implemented in [src/model.py](src/model.py) is:

1. Parse CLI args into a `TrainConfig`.
2. Set random seeds for reproducibility.
3. Load dataframe and normalize expected feature columns.
4. Detect supervision column (if present) or use weak supervision labels.
5. Build `BugExample` objects.
6. Split into train/validation sets with stratification when possible.
7. Build vocabulary from training text.
8. Build PyTorch datasets and dataloaders.
9. Optionally search hyperparameters using Dragonfly optimizer.
10. Train model for configured epochs.
11. Track best validation F1 and retain best state.
12. Save checkpoint to a timestamped or user-provided path.

## 3. Data and Feature Engineering

### 3.1 Input Data Expectations

The default dataset is [dataset/corpus_features.csv](dataset/corpus_features.csv).

The script can work with multiple CSV formats. If columns are missing, it derives fallback columns:

- `textual_features` from `Summary + Description`.
- `contextual_features` from `Description`.

### 3.2 Label Resolution

Label column is auto-detected from candidates:

- `label`
- `target`
- `is_duplicate`
- `duplicate`
- `bug_label`

If no explicit label exists, weak supervision is used from duplicate-related phrases such as "duplicate of" and "resolved as duplicate".

### 3.3 Feature Branches

#### Text branch

- Tokenizer: lowercase regex tokenizer `[a-z0-9_]+`.
- Vocabulary: built from training set frequencies with max size.
- Sequence handling: fixed `max_seq_len` by truncation/padding.
- Outputs: `input_ids`, `attention_mask`.

#### Temporal branch

- Parses date/time from known bug report formats.
- Uses cyclic encoding for periodic time components.
- Returns 9 floats:

1. `year_scaled`
2. `sin(month)`
3. `cos(month)`
4. `sin(weekday)`
5. `cos(weekday)`
6. `sin(hour)`
7. `cos(hour)`
8. `minute_scaled`
9. `is_weekend`

#### Categorical branch

- Uses structured payload when available, otherwise infers from text.
- Label spaces:

1. Priority: 5 classes
2. Severity: 5 classes
3. Issue type: 8 classes
4. Component: 11 classes

- One-hot concatenation size: 29 floats.

#### Contextual branch

- Extracts binary cues from contextual text:

1. stack trace presence
2. error/exception presence
3. code-like pattern presence
4. normalized token length

- Vector size: 4 floats.

## 4. Core Data Structures

### `BugExample`

Container for one report:

- `text: str`
- `temporal_vector: List[float]`
- `categorical_vector: List[float]`
- `contextual_vector: List[float]`
- `label: int`

### `Vocabulary`

Responsibilities:

1. Build token-to-id map from train text.
2. Encode text into fixed-length ids.
3. Generate attention mask.

Special tokens:

- `<pad>` id 0
- `<unk>` id 1

### `BugReportDataset`

Each sample yields:

- `input_ids`
- `attention_mask`
- `temporal_features`
- `categorical_features`
- `contextual_features`
- `label`
- `tag_ids`

`tag_ids` is a token-level broadcast of report label for CRF training (masked on padding positions).

## 5. Model Architecture: `ADLMBiLSTMCRF`

The model in [src/model.py](src/model.py) combines text and side channels:

1. Embedding layer over token ids.
2. BiLSTM text encoder.
3. Side encoder MLP for concatenated non-text features.
4. Fusion by concatenating text state and side representation.
5. Emission head producing token logits.
6. CRF for sequence-level decoding and training objective.

### Default dimensions (from CLI defaults)

- Embedding dim: 128
- LSTM hidden size: 128 (bidirectional output size 256)
- Side branch output size: 256
- Fusion size: 512
- Labels: 2 (duplicate/non-duplicate)

### Report-level decision rule

CRF outputs token tags. They are collapsed to one report prediction by:

- `positive_ratio = sum(tags) / len(tags)`
- report prediction = 1 if `positive_ratio >= 0.5`, else 0

### Loss

- Base loss: negative mean CRF log-likelihood
- Optional additive L2 penalty on CRF parameters (`crf_regularization`)

## 6. Training and Evaluation

### 6.1 Train loop (`train_one_epoch`)

1. Move batch to device.
2. Forward pass with `tag_ids`.
3. Backprop on `output["loss"]`.
4. Gradient clipping with max norm 1.0.
5. Optimizer step.

### 6.2 Eval loop (`evaluate`)

- Runs without gradient tracking.
- Computes loss (when available) and report-level predictions.
- Computes metrics:

1. Accuracy
2. Precision
3. Recall
4. F1
5. Cohen's Kappa
6. Mean validation loss

### 6.3 Split behavior (`split_examples`)

- Uses `train_test_split`.
- Stratifies only if both classes exist and each has at least 2 samples.

## 7. Dragonfly Hyperparameter Optimization

Enabled with `--use_dragonfly`.

Searches:

1. learning rate
2. hidden size
3. dropout
4. CRF regularization

Bounds:

- learning rate: [1e-4, 5e-3]
- hidden size: [48, 256] then rounded to nearest multiple of 8
- dropout: [0.1, 0.6]
- CRF regularization: [1e-6, 5e-3]

Fitness objective:

- Validation F1 after short inner training (`dragonfly_inner_epochs`).

## 8. Checkpoint Format

Saved checkpoint contains:

1. `model_state_dict`
2. `best_model_state_dict`
3. `vocabulary`
4. `model_config`
5. `best_metrics`
6. `label_column`
7. `label_mode`
8. `training_status`
9. `last_completed_epoch`
10. `epochs_requested`

Default output path behavior:

- If `--model_out` omitted, a timestamped file is generated under [models](models).

## 9. CLI Reference

Main entrypoint in [src/model.py](src/model.py):

```bash
python src/model.py [options]
```

Important options and defaults:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `dataset/corpus_features.csv` | Training CSV path |
| `--model_out` | auto timestamp under `models/` | Checkpoint output path |
| `--label_column` | `None` | Explicit label column name |
| `--max_vocab_size` | `15000` | Max vocabulary size |
| `--max_seq_len` | `128` | Token sequence length |
| `--embedding_dim` | `128` | Embedding dimension |
| `--hidden_size` | `128` | BiLSTM hidden size |
| `--dropout` | `0.30` | Dropout rate |
| `--learning_rate` | `1e-3` | Optimizer learning rate |
| `--crf_regularization` | `1e-4` | CRF L2 penalty weight |
| `--batch_size` | `16` | Training batch size |
| `--epochs` | `8` | Number of epochs |
| `--val_size` | `0.2` | Validation split ratio |
| `--seed` | `42` | Random seed |
| `--use_dragonfly` | flag | Enable hyperparameter search |
| `--dragonfly_population` | `5` | Dragonfly population |
| `--dragonfly_iterations` | `2` | Dragonfly iterations |
| `--dragonfly_inner_epochs` | `2` | Inner epochs per candidate |

## 10. Example Commands

Train with defaults:

```bash
python src/model.py
```

Train with explicit label column and custom output:

```bash
python src/model.py --label_column label --model_out models/adlm_custom.pt
```

Train with Dragonfly tuning:

```bash
python src/model.py --use_dragonfly --dragonfly_population 8 --dragonfly_iterations 4
```

## 11. Operational Notes

1. If the dataset has no explicit label column, training still runs using weak supervision labels.
2. A minimum of 10 valid examples is required; otherwise training raises an error.
3. On `KeyboardInterrupt`, the script saves an interrupted checkpoint instead of losing progress.
4. Device is selected automatically: CUDA if available, otherwise CPU.
