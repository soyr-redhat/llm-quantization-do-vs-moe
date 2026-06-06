# LLM Quantization Exercise: Decoder-Only vs. Mixture of Experts

Comparing FP8, INT8, INT4, and FP4 quantization schemes across two model architectures using [llm-compressor](https://github.com/vllm-project/llm-compressor).

## Models

| Model | Architecture | Total Params | Active Params |
|-------|-------------|-------------|---------------|
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Decoder-only | 1B | 1B |
| [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) | Mixture of Experts | ~14B | 2.7B |

## Quantization Schemes

All schemes use GPTQ with 512 calibration samples from `ultrachat_200k`.

| Scheme | Format | Description |
|--------|--------|-------------|
| W8A8 | INT8 | 8-bit integer weights and activations |
| W4A16 | INT4 | 4-bit integer weights, 16-bit activations |
| FP8 | FP8 | 8-bit floating point weights and activations |
| NVFP4 | FP4 | 4-bit floating point weights (Blackwell/Hopper) |

## Project Structure

```
src/
  main.py               # Quantization script (GPTQ across all schemes)
  tokenize_dataset.py    # Calibration dataset loading and tokenization
  generations.py         # Sample generation testing
openshift/
  quantize-job.yaml      # K8s job for running quantization on GPU cluster
  quantize-pvc.yaml      # PVC for storing quantized model output
report.tex               # LaTeX report with results and analysis
```

## Hardware

- Quantization: 1xH200
- Inference testing: NVIDIA RTX 3090 Turbo
    - for applicable models
