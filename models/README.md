# Models

Put base checkpoints and Diffusers model directories here. LoRA-kit tracks this README, but ignores downloaded model files so large weights do not end up in git.

For example, download Stable Diffusion 1.5 with the Hugging Face CLI:

```bash
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  v1-5-pruned-emaonly.safetensors \
  --local-dir LoRA-kit/models
```

Then reference it by local name when training:

```bash
lorakit train fox-solo --model v1-5-pruned-emaonly
```

LoRA-kit also accepts explicit paths:

```bash
lorakit train fox-solo --model models/v1-5-pruned-emaonly.safetensors
```
