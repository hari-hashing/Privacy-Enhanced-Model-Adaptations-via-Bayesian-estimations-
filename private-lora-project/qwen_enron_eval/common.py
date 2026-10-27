import glob
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- edit these to match how your dataset / training prompt was built ----
TEXT_COL = "text"
LABEL_COL = "label"
LABEL_WORDS = {0: "ham", 1: "spam"}
PROMPT_COL = "prompt_ids"
PREPARED_LABEL_COL = "class_label"
SYSTEM_PROMPT = "Classify the email as spam or ham. Answer with one word."
MAX_CHARS = 2000  # long emails are cut to this many characters


def _resolve_data_path(data_dir):
    data_path = Path(data_dir).expanduser()
    if not data_path.is_absolute():
        working_directory_path = data_path.resolve()
        project_path = Path(__file__).resolve().parents[1] / data_path
        if working_directory_path.exists():
            data_path = working_directory_path
        elif project_path.exists():
            data_path = project_path
        else:
            data_path = working_directory_path
    return data_path


def _load_saved_split(data_path, split):
    data_dir = str(data_path)
    split_dir = data_path / split
    if (data_path / "dataset_dict.json").is_file():
        dataset = load_from_disk(data_dir)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected a Hugging Face DatasetDict at {data_dir}")
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in dataset at {data_dir}")
        return dataset[split], data_dir
    if (split_dir / "state.json").is_file():
        return load_from_disk(str(split_dir)), str(split_dir)
    if (data_path / "state.json").is_file():
        return load_from_disk(data_dir), data_dir
    return None, None


def load_split(data_dir, split):
    """Load a raw split from flat files or a saved Hugging Face dataset."""
    data_path = _resolve_data_path(data_dir)
    data_dir = str(data_path)
    files = glob.glob(os.path.join(data_dir, f"{split}.*"))
    if files:
        path = files[0]
        if path.endswith(".jsonl"):
            df = pd.read_json(path, lines=True)
        elif path.endswith(".json"):
            df = pd.read_json(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
    else:
        dataset, dataset_path = _load_saved_split(data_path, split)
        if dataset is None:
            raise FileNotFoundError(
                f"No '{split}.*' file or saved Hugging Face split found in {data_dir}"
            )

        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected a Hugging Face Dataset at {dataset_path}")
        missing_columns = {TEXT_COL, LABEL_COL}.difference(dataset.column_names)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            available = ", ".join(dataset.column_names)
            raise ValueError(
                f"Dataset split '{split}' at {dataset_path} is missing required "
                f"raw columns: {missing}. Available columns: {available}. "
                "MIA requires raw email text and labels; use the source dataset "
                "(for example, dataset/enron_spam) instead of the prepared "
                "tokenized dataset."
            )
        df = dataset.select_columns([TEXT_COL, LABEL_COL]).to_pandas()

    missing_columns = {TEXT_COL, LABEL_COL}.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Split '{split}' is missing required columns: {missing}")
    df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str[:MAX_CHARS]
    df[LABEL_COL] = df[LABEL_COL].apply(
        lambda x: 1 if str(x).strip().lower() in ("1", "spam") else 0
    )
    return df.reset_index(drop=True)


def load_mia_split(data_dir, split):
    """Load a raw or prepared split for membership inference."""
    data_path = _resolve_data_path(data_dir)
    if glob.glob(str(data_path / f"{split}.*")):
        return load_split(data_path, split)

    dataset, _ = _load_saved_split(data_path, split)
    if isinstance(dataset, Dataset) and {
        PROMPT_COL,
        PREPARED_LABEL_COL,
        "input_ids",
        "labels",
    }.issubset(dataset.column_names):
        return dataset
    return load_split(data_path, split)


def load_prepared_split(data_dir, split="test"):
    """Load a prepared Hugging Face split from a DatasetDict or split directory."""
    data_path = _resolve_data_path(data_dir)
    dataset, dataset_path = _load_saved_split(data_path, split)
    if dataset is None:
        raise FileNotFoundError(f"No saved Hugging Face split '{split}' found at {data_path}")
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Hugging Face Dataset at {dataset_path}")

    required_columns = {PROMPT_COL, PREPARED_LABEL_COL}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Prepared {split} split at {dataset_path} is missing required columns: {missing}"
        )
    return dataset


def load_model(adapter_path):
    """Load the adapter's base model and attach the saved PEFT adapter."""
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    base_model_path = adapter_config.base_model_name_or_path
    if not base_model_path:
        raise ValueError(f"Adapter at {adapter_path} does not declare a base model")

    tok = AutoTokenizer.from_pretrained(base_model_path)
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.to(device).eval()
    return model, tok


def build_prompt(tok, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )