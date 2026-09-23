# Private LoRA training

This scaffold supports standard PEFT LoRA (`dp_lora`) over Hugging Face sequence
classifiers. Opacus provides
per-example gradient sampling, ghost clipping, and the Gaussian mechanism for
DP-SGD.

```bash
python -m pip install -e .
private-lora-train --config config/example.yaml
```

Before reporting privacy guarantees, account for sampling, composition, and the
actual number of optimizer steps with an accountant appropriate to your release.
The example is intentionally small and is not a production privacy budget.
