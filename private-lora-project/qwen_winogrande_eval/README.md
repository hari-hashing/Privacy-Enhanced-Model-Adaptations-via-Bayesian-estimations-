# Qwen3 Winogrande-M evaluation

This directory prepares the Hugging Face `allenai/winogrande` dataset's `winogrande_m`
configuration for the same `Qwen/Qwen3-1.7B` causal-LM and QLoRA/DP-SGD
training path used by the Enron evaluation. Each example asks Qwen to choose
option A or B; only the answer tokens contribute to the training loss.

Model weights and checkpoints, including `.safetensors` files, must not be
committed or pushed to Git. The repository `.gitignore` excludes these files.

## Prepare data and train DP-LoRA

Run these commands from `private-lora-project/` in the project's CUDA-enabled
environment. The DP trainer uses two GPUs and the same rank, alpha, clipping,
noise, and optimizer settings as the Qwen Enron run.

```bash
python qwen_winogrande_eval/prepare_data.py \
  --output-dir dataset/winogrande_m_qwen3
torchrun --standalone --nproc_per_node=2 -m private_lora.finetune \
  --config qwen_winogrande_eval/config/dp_lora.yaml
```

The trainer saves its per-step loss history and plot alongside the adapter.
Winogrande's official test answers are withheld, so reported utility metrics
use the labeled `validation` split.

## Evaluate adapters and membership inference

The evaluation accepts any compatible PEFT adapter directory, including the
DP-LoRA output and a Bayesian-LoRA adapter trained elsewhere. Each method
should have a separate run directory so its results are not overwritten.

```bash
python qwen_winogrande_eval/eval.py \
  --adapter-path outputs/winogrande-m-qwen3-1.7b-dp-lora \
  --data-dir dataset/winogrande_m_qwen3
python qwen_winogrande_eval/MIA.py \
  --model-path outputs/winogrande-m-qwen3-1.7b-dp-lora \
  --data-dir dataset/winogrande_m_qwen3
```

Evaluation writes accuracy, precision, recall, F1, ROC AUC, Brier score, and
15-bin expected calibration error (ECE), plus confusion-matrix, ROC, score
distribution, and calibration plots. MIA compares a seeded sample of training
members with validation non-members and reports loss, Min-K, zlib, and answer
loss ROC AUCs; plots include the MIA ROC curves and member/non-member loss
distribution.

To evaluate a second adapter, pass its path to the same two scripts. Then
aggregate the results (default output directories are based on adapter name):

```bash
python qwen_winogrande_eval/aggregate.py \
  --run dp_lora=qwen_winogrande_eval/results/winogrande-m-qwen3-1.7b-dp-lora \
  --run bayesian_lora=qwen_winogrande_eval/results/bayesian_adapter_name
```

Replace `bayesian_adapter_name` with the Bayesian adapter's directory name.
The aggregate output contains per-method CSV rows and the across-method mean
and standard deviation for ECE, utility metrics, MIA AUC, and low-FPR TPR.
