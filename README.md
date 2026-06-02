# PolySpeech-100 Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-Extended_Version-B31B1B.svg)]([https://arxiv.org/abs/2606.01016])
[![Demo](https://img.shields.io/badge/Demo-Interactive_Map-success.svg)](https://youngseng.github.io/PolySpeech-100/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Full_Results-yellow.svg)](https://youngseng.github.io/PolySpeech-100/benchmark.html)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging_Face-blue.svg)](https://huggingface.co/datasets/youngseng/PolySpeech-100-v1)

<!-- The dataset is hosted on Hugging Face at [`youngseng/PolySpeech-100-v1`](https://huggingface.co/datasets/youngseng/PolySpeech-100-v1) and contains compressed Parquet files to store audio and text data. -->

Welcome to the **PolySpeech-100** Benchmark!

---

## 🏆 Leaderboard

Here is a quick glance at the top-performing models on the PolySpeech-100 benchmark. 

| Model | Type | Overall | High-Res | CN-Dialect | Low-Res |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini-3-flash** | Closed-Source | **85.30** | **94.26** | **83.54** | **84.61** |
| **GPT-Audio-mini** | Closed-Source | 56.63 | 83.56 | 55.58 | 53.56 |
| **Whisper-v3 + Qwen2.5** | Pipeline | 53.86 | 83.74 | 62.62 | 48.12 |
| **Fun-Audio-Chat** | Open E2E (st2t) | 52.88 | 84.82 | 77.06 | 43.26 |

## Setup & Installation

```bash
git clone https://github.com/YoungSeng/PolySpeech-100
cd PolySpeech-100

# Create and activate conda environment
conda create -n PolySpeech-100-Env python=3.10 -y
conda activate PolySpeech-100-Env

# Install core dependencies
pip install pandas pyarrow tqdm huggingface_hub librosa soundfile torch transformers scipy
```

## Dataset Preparation

The PolySpeech-100 dataset is massive (~430 GB). To save bandwidth and time, we recommend downloading a specific language (e.g., English) for testing before downloading the entire benchmark.

We provide a unified script to automatically download the dataset from Hugging Face and restore the original `.wav` and `.txt` files.

(If you are experiencing slow download speeds (e.g. in Mainland China) from Hugging Face, you can use the official mirror site. Simply set the `HF_ENDPOINT` environment variable before running the script.)

Example 1: Download and restore a specific language (Recommended for quick start)
```bash
# (optional) export HF_ENDPOINT=https://hf-mirror.com
python prepare_dataset.py --lang eng_Latn --output_dir ./Restored-PolySpeech
```

Example 2: Download and restore the entire dataset
```bash
python prepare_dataset.py --lang all --output_dir ./Restored-PolySpeech
```

[//]: # (This will create a ./Restored-PolySpeech directory containing the uncompressed dataset folders &#40;e.g., `./Restored-PolySpeech/lang=eng_Latn`&#41;.)

## Running Inference

<!-- Since each model relies on its own custom architecture and local files, -->
The inference scripts provided below are evaluation wrappers. You must manually configure the environments and download the pre-trained model weights from their respective official repositories.

### st2s  (Qwen2.5-Omni Example)

Before running, ensure you have the specific utility dependencies installed (like `flash-attn`, or uncomment the `attn_implementation="flash_attention_2"` in the script).

<!-- pip install qwen-omni-utils[decord] -U torchvision accelerate -->
<!-- pip install -U flash-attn --no-build-isolation -->

1. Basic Zero-shot Evaluation (Clean Audio):
```bash
# Evaluate English only
CUDA_VISIBLE_DEVICES=0 python inference_qwen2_5_omni.py --lang eng_Latn --save_audio --overwrite
# entire dataset
# CUDA_VISIBLE_DEVICES=0 python inference_qwen2_5_omni.py --lang all --save_audio
```

2. High Noise Environment Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_qwen2_5_omni.py --lang eng_Latn --shots 0 --augment noise_high --save_audio
```

3. Chain-of-Thought (CoT) + 3-shot Evaluation:

```bash
# Zero-shot with Chain-of-Thought
CUDA_VISIBLE_DEVICES=0 python inference_qwen2_5_omni.py --lang eng_Latn --cot --save_audio
# 3-shot prompt evaluation
# CUDA_VISIBLE_DEVICES=0 python inference_qwen2_5_omni.py --lang eng_Latn --shots 3 --save_audio
```



### s2s  (Covo-Audio Example)

To run this evaluation, you must first clone and follow the official `Covo-audio` [repository](https://github.com/Tencent/Covo-Audio)  to configure and run the script from within their directory.

```
cd Covo-Audio
CUDA_VISIBLE_DEVICES=0 python inference_covoaudio.py --lang eng_Latn --model_dir ./covoaudio --save_audio
```

## Evaluation
After generating the inference results, you can calculate the accuracy and generate summary reports using the evaluation script:
```
python evaluate.py
```

## Citation

If you find this benchmark useful for your research, please consider citing our paper:

```
@misc{yang2026polyspeech100largescalebenchmarkspeech,
      title={PolySpeech-100: A Large-Scale Benchmark for Speech Understanding Across 100+ Languages and Dialects}, 
      author={Sicheng Yang and Shulan Ruan and Shiwei Wu and Yu Liu and Lu Fan and Zhi Li and You He},
      year={2026},
      eprint={2606.01016},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.01016}, 
}
```
