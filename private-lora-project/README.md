# Qwen3 DP-LoRA on Enron Spam

This project fine-tunes `Qwen/Qwen3-14B` as a binary Enron email classifier. It uses 4-bit NF4 quantization, LoRA on `q_proj` and `v_proj`, gradient checkpointing, FSDP across two GPUs, and FSDP CPU parameter offload. DP-SGD clips the LoRA gradient separately for each sampled email and adds Gaussian noise; the RDP accountant reports an approximate epsilon for the configured noise multiplier and sampling rate.

## Install and prepare data

Install the project dependencies in an environment with a CUDA-enabled PyTorch build and a compatible bitsandbytes installation:

```bash
pip install -e .
```

Download the existing SetFit Enron spam dataset, then convert each email and its ham/spam label into the causal-LM format used by training:

```bash
download-enron --output dataset/enron_spam
private-lora-prepare-data --config config/example.yaml
```

To verify the model-loading path independently, including FSDP and CPU offload, run:

```bash
torchrun --standalone --nproc_per_node=2 -m private_lora.load_model --config config/example.yaml --fsdp
```

The prepared dataset contains train, validation, and test splits. Only the output label tokens contribute to the language-model loss; the email and instruction tokens are masked with `-100`.

## Train

Run from this project directory with exactly two visible GPUs:

```bash
torchrun --standalone --nproc_per_node=2 -m private_lora.finetune --config config/example.yaml
```

Each FSDP rank loads a quantized model shard and processes the same Poisson-sampled email indices. This is deliberate: the ranks shard model parameters, not examples, so per-email gradient norms can be combined across parameter shards before clipping. This makes DP accumulation substantially slower than ordinary LoRA training. The sampler/accountant use `expected_batch_size / training_set_size` as the Poisson sampling rate.

## Evaluate

After training, evaluate exact ham/spam predictions on the held-out test split:

```bash
private-lora-evaluate --config config/example.yaml --adapter outputs/enron-qwen3-14b-dp-lora
```

The evaluator loads the same 4-bit base model and the saved adapter, then prints accuracy, precision, recall, and F1.

## Hardware and privacy notes

FSDP wraps the Qwen3 decoder layers with `use_orig_params=True` and CPU parameter offload. NF4 uses BF16 quantization storage for FSDP compatibility. `device_map=auto` is intentionally not used because it conflicts with FSDP ownership. The model must fit at least its active FSDP shard and activations on each GPU; CPU offload does not eliminate activation memory. Email-level clipping protects each sampled training email, but does not by itself protect against data already exposed in downloaded/prepared files or logs. The reported epsilon is an accounting estimate, not a guarantee independent of the configured sampling and adjacency assumptions.

The current model loader requires CUDA, two available GPUs, and `bitsandbytes`; the training process raises a clear error when those prerequisites or the expected FSDP world size are missing.

