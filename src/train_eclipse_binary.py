from model_compatible_with_colab import run_colab


CONFIG_OVERRIDES = {
    "data_path": "dataset/eclipse_platform.csv",
    "prepared_data_out": "dataset/eclipse_platform_binary_balanced_preprocessed.csv",
    "prepare_eclipse_binary": True,
    "negative_to_positive_ratio": 1.0,
    "label_column": "is_duplicate",
    "model_out": "models/adlm_eclipse_binary_balanced_gpu_e5.pt",
    "epochs": 5,
    "batch_size": 128,
    "max_seq_len": 128,
    "max_vocab_size": 20000,
    "embedding_dim": 128,
    "hidden_size": 128,
    "dropout": 0.30,
    "learning_rate": 0.0015,
    "val_size": 0.2,
    "seed": 42,
    "num_workers": 0,
    "prefetch_factor": 2,
    "persistent_workers": False,
    "pin_memory": True,
    "use_amp": True,
    "gradient_accumulation_steps": 1,
    "use_weighted_sampler": True,
    "use_dragonfly": False,
}


if __name__ == "__main__":
    run_colab(CONFIG_OVERRIDES)
