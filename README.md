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

```in the future, additional quantization algorithms can be explored```

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


## Example Outputs

| Model | Scheme | Format | Input Prompt | Output |
|-------|--------|--------|--------------|--------|
|[Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)|W4A16| INT4 | `Explain quantization in machine learning in 3 sentences.` | `Quantization in machine learning refers to the process of representing continuous features or inputs in discrete or numerical form, such as integers or floats, to reduce the number of dimensions and improve model efficiency. This is typically done using techniques like discrete wavelet transform or quantization layers, where the continuous input is discretized into discrete values. The resulting discrete representations can be used as input features to the model, allowing for faster and more efficient training of machine learning models.` |
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)| W8A8 | INT8 | `Explain quantization in machine learning in 3 sentences.` | "" | 
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)| FP8 | INT8 | `Explain quantization in machine learning in 3 sentences.` | "" | 



| Model | Scheme | Format | Input Prompt | Output |
|-------|--------|--------|--------------|--------|
|[Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)|