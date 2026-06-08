# LLM Quantization Exercise: Decoder-Only vs. Mixture of Experts

Comparing FP8, INT8, INT4, and FP4 quantization schemes across two model architectures using [llm-compressor](https://github.com/vllm-project/llm-compressor).

## Models

| Model | Architecture | Total Params | Active Params |
|-------|-------------|-------------|---------------|
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Decoder-only | 1B | 1B |
| [DeepSeek-MoE-16B-Chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) | Mixture of Experts | ~16B | 2.8B |

## Quantization Schemes

All schemes use GPTQ with 512 calibration samples from `ultrachat_200k`.

| Scheme | Format | Description |
|--------|--------|-------------|
| W8A8 | INT8 | 8-bit integer weights and activations |
| W4A16 | INT4 | 4-bit integer weights, 16-bit activations |
| FP8 | FP8 | 8-bit floating point weights and activations |
| NVFP4 | FP4 | 4-bit floating point weights and activations (Blackwell/Hopper) |

## Project Structure

```
src/
  main.py               # Quantization script (GPTQ across all schemes)
  tokenize_dataset.py   # Calibration dataset loading and tokenization
  metrics.py            # Chart generation and CSV export from benchmark results
  benchmark.sh          # Local benchmarking script (lm-eval + guidellm)
openshift/
  quantize-job.yaml     # K8s job for running quantization on GPU cluster
  quantize-pvc.yaml     # PVC for storing quantized model output
  benchmark-job.yaml    # K8s job for running benchmarks on GPU cluster
  inference-deployment.yaml  # vLLM inference deployment
  inference-pvc.yaml    # PVC for inference model storage
report.tex              # LaTeX report with results and analysis
```


## Example Outputs

| Model | Scheme | Format | Input Prompt | Output |
|-------|--------|--------|--------------|--------|
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)| W8A8 | INT8 | `Explain quantization in machine learning in 3 sentences.` | `Quantization in machine learning is a process where the continuous input data is discretized or converted into a discrete set of values, often represented as integers or binary numbers, to reduce the number of parameters and computational requirements. This technique is commonly used in neural networks and deep learning models to improve efficiency and reduce memory usage. By quantizing the input data, the model can learn to represent complex patterns and relationships in the data with fewer parameters, leading to improved performance and reduced computational complexity.` | 
|[Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)|W4A16| INT4 | `Explain quantization in machine learning in 3 sentences.` | `Quantization in machine learning refers to the process of representing continuous features or inputs in discrete or numerical form, such as integers or floats, to reduce the number of dimensions and improve model efficiency. This is typically done using techniques like discrete wavelet transform or quantization layers, where the continuous input is discretized into discrete values. The resulting discrete representations can be used as input features to the model, allowing for faster and more efficient training of machine learning models.` |
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)| FP8 | FP8 | `Explain quantization in machine learning in 3 sentences.` | `Quantization in machine learning involves reducing the precision of numerical data, such as weights, biases, or feature values, to a lower number of bits to improve model efficiency and reduce memory requirements. This is typically achieved by using techniques like quantization-aware neural networks, which adapt the model's architecture to accommodate the reduced precision. By quantizing the model's parameters, machine learning algorithms can reduce computation time, memory usage, and energy consumption, making them more suitable for deployment in resource-constrained environments.` | 
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)| NVFP4 | FP4 | `Explain quantization in machine learning in 3 sentences.` | `Quantization in machine learning is a process where numerical values are discretized into a discrete set of possible values, typically using quantization schemes such as integer or floating-point quantization, to reduce the number of possible inputs and improve model interpretability. This approach is often used in deep learning models, particularly in neural networks, to reduce computational requirements, increase model efficiency, and improve performance on certain datasets. Quantization can also be used to mitigate the effects of numerical instability and improve the robustness of machine learning models to outliers and noisy data.` | 



| Model | Scheme | Format | Input Prompt | Output |
|-------|--------|--------|--------------|--------|
|[DeepSeek-MoE-16B-Chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)| W8A8 | INT8 |`Explain quantization in machine learning in 3 sentences.` | |
|[DeepSeek-MoE-16B-Chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)| W4A16 | INT4|`Explain quantization in machine learning in 3 sentences.` | |
|[DeepSeek-MoE-16B-Chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)| FP8 | FP8 |`Explain quantization in machine learning in 3 sentences.` | |
|[DeepSeek-MoE-16B-Chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)| NVFP4 | FP4 |`Explain quantization in machine learning in 3 sentences.` | |
