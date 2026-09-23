# Enron email dataset

The example DP-LoRA configuration uses the Hugging Face
`SetFit/enron_spam` dataset. It contains labeled Enron email text for binary
spam classification and provides `train` and `test` splits with these columns:

- `text`: email body and metadata
- `label`: `0` for ham and `1` for spam

Download the dataset once before training:

```bash
download-enron
```

The command saves the `train` and `test` splits to `dataset/enron_spam`.
Choose another location with `--output`:

```bash
download-enron --output /path/to/enron_spam
```

Training reads `dataset_path` from `config/example.yaml` using
`load_from_disk`; it does not download data while training. The checked-in
folder contains documentation only because dataset files are large and may
have separate usage terms.