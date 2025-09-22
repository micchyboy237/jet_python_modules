---
quantized_by: bartowski
pipeline_tag: text-generation
widget:
- example_title: Hermes 3
  messages:
  - role: system
    content: You are a sentient, superintelligent artificial general intelligence,
      here to teach and assist me.
  - role: user
    content: Write a short story about Goku discovering kirby has teamed up with Majin
      Buu to destroy the world.
language:
- en
base_model: NousResearch/Hermes-3-Llama-3.2-3B
tags:
- Llama-3
- instruct
- finetune
- chatml
- gpt4
- synthetic data
- distillation
- function calling
- json mode
- axolotl
- roleplaying
- chat
license: llama3
model-index:
- name: Hermes-3-Llama-3.1-405B
  results: []
---

## Llamacpp imatrix Quantizations of Hermes-3-Llama-3.2-3B

Using <a href="https://github.com/ggerganov/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggerganov/llama.cpp/releases/tag/b4273">b4273</a> for quantization.

Original model: https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B

All quants made using imatrix option with dataset from [here](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8)

Run them in [LM Studio](https://lmstudio.ai/)

## Prompt format

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

## Download a file (not the whole branch) from below:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [Hermes-3-Llama-3.2-3B-f32.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-f32.gguf) | f32 | 12.86GB | false | Full F32 weights. |
| [Hermes-3-Llama-3.2-3B-f16.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-f16.gguf) | f16 | 6.43GB | false | Full F16 weights. |
| [Hermes-3-Llama-3.2-3B-Q8_0.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q8_0.gguf) | Q8_0 | 3.42GB | false | Extremely high quality, generally unneeded but max available quant. |
| [Hermes-3-Llama-3.2-3B-Q6_K_L.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q6_K_L.gguf) | Q6_K_L | 2.74GB | false | Uses Q8_0 for embed and output weights. Very high quality, near perfect, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q6_K.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q6_K.gguf) | Q6_K | 2.64GB | false | Very high quality, near perfect, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q5_K_L.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q5_K_L.gguf) | Q5_K_L | 2.42GB | false | Uses Q8_0 for embed and output weights. High quality, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q5_K_M.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q5_K_M.gguf) | Q5_K_M | 2.32GB | false | High quality, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q5_K_S.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q5_K_S.gguf) | Q5_K_S | 2.27GB | false | High quality, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q4_K_L.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q4_K_L.gguf) | Q4_K_L | 2.11GB | false | Uses Q8_0 for embed and output weights. Good quality, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q4_K_M.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q4_K_M.gguf) | Q4_K_M | 2.02GB | false | Good quality, default size for most use cases, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q4_K_S.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q4_K_S.gguf) | Q4_K_S | 1.93GB | false | Slightly lower quality with more space savings, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q4_0_8_8.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q4_0_8_8.gguf) | Q4_0_8_8 | 1.92GB | false | Optimized for ARM and AVX inference. Requires 'sve' support for ARM (see details below). *Don't use on Mac*. |
| [Hermes-3-Llama-3.2-3B-Q4_0_4_8.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q4_0_4_8.gguf) | Q4_0_4_8 | 1.92GB | false | Optimized for ARM inference. Requires 'i8mm' support (see details below). *Don't use on Mac*. |
| [Hermes-3-Llama-3.2-3B-Q4_0_4_4.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q4_0_4_4.gguf) | Q4_0_4_4 | 1.92GB | false | Optimized for ARM inference. Should work well on all ARM chips, not for use with GPUs. *Don't use on Mac*. |
| [Hermes-3-Llama-3.2-3B-Q4_0.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q4_0.gguf) | Q4_0 | 1.92GB | false | Legacy format, offers online repacking for ARM CPU inference. |
| [Hermes-3-Llama-3.2-3B-IQ4_NL.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-IQ4_NL.gguf) | IQ4_NL | 1.92GB | false | Similar to IQ4_XS, but slightly larger. Offers online repacking for ARM CPU inference. |
| [Hermes-3-Llama-3.2-3B-Q3_K_XL.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q3_K_XL.gguf) | Q3_K_XL | 1.91GB | false | Uses Q8_0 for embed and output weights. Lower quality but usable, good for low RAM availability. |
| [Hermes-3-Llama-3.2-3B-IQ4_XS.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-IQ4_XS.gguf) | IQ4_XS | 1.83GB | false | Decent quality, smaller than Q4_K_S with similar performance, *recommended*. |
| [Hermes-3-Llama-3.2-3B-Q3_K_L.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q3_K_L.gguf) | Q3_K_L | 1.82GB | false | Lower quality but usable, good for low RAM availability. |
| [Hermes-3-Llama-3.2-3B-Q3_K_M.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q3_K_M.gguf) | Q3_K_M | 1.69GB | false | Low quality. |
| [Hermes-3-Llama-3.2-3B-IQ3_M.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-IQ3_M.gguf) | IQ3_M | 1.60GB | false | Medium-low quality, new method with decent performance comparable to Q3_K_M. |
| [Hermes-3-Llama-3.2-3B-Q3_K_S.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q3_K_S.gguf) | Q3_K_S | 1.54GB | false | Low quality, not recommended. |
| [Hermes-3-Llama-3.2-3B-IQ3_XS.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-IQ3_XS.gguf) | IQ3_XS | 1.48GB | false | Lower quality, new method with decent performance, slightly better than Q3_K_S. |
| [Hermes-3-Llama-3.2-3B-Q2_K_L.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q2_K_L.gguf) | Q2_K_L | 1.46GB | false | Uses Q8_0 for embed and output weights. Very low quality but surprisingly usable. |
| [Hermes-3-Llama-3.2-3B-Q2_K.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q2_K.gguf) | Q2_K | 1.36GB | false | Very low quality but surprisingly usable. |
| [Hermes-3-Llama-3.2-3B-IQ2_M.gguf](https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-IQ2_M.gguf) | IQ2_M | 1.23GB | false | Relatively low quality, uses SOTA techniques to be surprisingly usable. |

## Embed/output weights

Some of these quants (Q3_K_XL, Q4_K_L etc) are the standard quantization method with the embeddings and output weights quantized to Q8_0 instead of what they would normally default to.

## Downloading using huggingface-cli

<details>
  <summary>Click to view download instructions</summary>

First, make sure you have hugginface-cli installed:

```
pip install -U "huggingface_hub[cli]"
```

Then, you can target the specific file you want:

```
huggingface-cli download bartowski/Hermes-3-Llama-3.2-3B-GGUF --include "Hermes-3-Llama-3.2-3B-Q4_K_M.gguf" --local-dir ./
```

If the model is bigger than 50GB, it will have been split into multiple files. In order to download them all to a local folder, run:

```
huggingface-cli download bartowski/Hermes-3-Llama-3.2-3B-GGUF --include "Hermes-3-Llama-3.2-3B-Q8_0/*" --local-dir ./
```

You can either specify a new local-dir (Hermes-3-Llama-3.2-3B-Q8_0) or download them all in place (./)

</details>

## Q4_0_X_X information

New: Thanks to efforts made to have online repacking of weights in [this PR](https://github.com/ggerganov/llama.cpp/pull/9921), you can now just use Q4_0 if your llama.cpp has been compiled for your ARM device.

Similarly, if you want to get slightly better performance, you can use IQ4_NL thanks to [this PR](https://github.com/ggerganov/llama.cpp/pull/10541) which will also repack the weights for ARM, though only the 4_4 for now. The loading time may be slower but it will result in an overall speed incrase.

<details>
  <summary>Click to view Q4_0_X_X information</summary>
These are *NOT* for Metal (Apple) or GPU (nvidia/AMD/intel) offloading, only ARM chips (and certain AVX2/AVX512 CPUs).

If you're using an ARM chip, the Q4_0_X_X quants will have a substantial speedup. Check out Q4_0_4_4 speed comparisons [on the original pull request](https://github.com/ggerganov/llama.cpp/pull/5780#pullrequestreview-21657544660)

To check which one would work best for your ARM chip, you can check [AArch64 SoC features](https://gpages.juszkiewicz.com.pl/arm-socs-table/arm-socs.html) (thanks EloyOn!).

If you're using a CPU that supports AVX2 or AVX512 (typically server CPUs and AMD's latest Zen5 CPUs) and are not offloading to a GPU, the Q4_0_8_8 may offer a nice speed as well:

<details>
  <summary>Click to view benchmarks on an AVX2 system (EPYC7702)</summary>

| model                          |       size |     params | backend    | threads |          test |                  t/s |  % (vs Q4_0)  |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |-------------: |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |      64 |         pp512 |        204.03 ± 1.03 |          100% |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |      64 |        pp1024 |        282.92 ± 0.19 |          100% |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |      64 |        pp2048 |        259.49 ± 0.44 |          100% |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |      64 |         tg128 |         39.12 ± 0.27 |          100% |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |      64 |         tg256 |         39.31 ± 0.69 |          100% |
| qwen2 3B Q4_0                  |   1.70 GiB |     3.09 B | CPU        |      64 |         tg512 |         40.52 ± 0.03 |          100% |
| qwen2 3B Q4_K_M                |   1.79 GiB |     3.09 B | CPU        |      64 |         pp512 |        301.02 ± 1.74 |          147% |
| qwen2 3B Q4_K_M                |   1.79 GiB |     3.09 B | CPU        |      64 |        pp1024 |        287.23 ± 0.20 |          101% |
| qwen2 3B Q4_K_M                |   1.79 GiB |     3.09 B | CPU        |      64 |        pp2048 |        262.77 ± 1.81 |          101% |
| qwen2 3B Q4_K_M                |   1.79 GiB |     3.09 B | CPU        |      64 |         tg128 |         18.80 ± 0.99 |           48% |
| qwen2 3B Q4_K_M                |   1.79 GiB |     3.09 B | CPU        |      64 |         tg256 |         24.46 ± 3.04 |           83% |
| qwen2 3B Q4_K_M                |   1.79 GiB |     3.09 B | CPU        |      64 |         tg512 |         36.32 ± 3.59 |           90% |
| qwen2 3B Q4_0_8_8              |   1.69 GiB |     3.09 B | CPU        |      64 |         pp512 |        271.71 ± 3.53 |          133% |
| qwen2 3B Q4_0_8_8              |   1.69 GiB |     3.09 B | CPU        |      64 |        pp1024 |       279.86 ± 45.63 |          100% |
| qwen2 3B Q4_0_8_8              |   1.69 GiB |     3.09 B | CPU        |      64 |        pp2048 |        320.77 ± 5.00 |          124% |
| qwen2 3B Q4_0_8_8              |   1.69 GiB |     3.09 B | CPU        |      64 |         tg128 |         43.51 ± 0.05 |          111% |
| qwen2 3B Q4_0_8_8              |   1.69 GiB |     3.09 B | CPU        |      64 |         tg256 |         43.35 ± 0.09 |          110% |
| qwen2 3B Q4_0_8_8              |   1.69 GiB |     3.09 B | CPU        |      64 |         tg512 |         42.60 ± 0.31 |          105% |

Q4_0_8_8 offers a nice bump to prompt processing and a small bump to text generation

</details>

</details>

## Which file should I choose?

<details>
  <summary>Click here for details</summary>

A great write up with charts showing various performances is provided by Artefact2 [here](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)

The first thing to figure out is how big a model you can run. To do this, you'll need to figure out how much RAM and/or VRAM you have.

If you want your model running as FAST as possible, you'll want to fit the whole thing on your GPU's VRAM. Aim for a quant with a file size 1-2GB smaller than your GPU's total VRAM.

If you want the absolute maximum quality, add both your system RAM and your GPU's VRAM together, then similarly grab a quant with a file size 1-2GB Smaller than that total.

Next, you'll need to decide if you want to use an 'I-quant' or a 'K-quant'.

If you don't want to think too much, grab one of the K-quants. These are in format 'QX_K_X', like Q5_K_M.

If you want to get more into the weeds, you can check out this extremely useful feature chart:

[llama.cpp feature matrix](https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix)

But basically, if you're aiming for below Q4, and you're running cuBLAS (Nvidia) or rocBLAS (AMD), you should look towards the I-quants. These are in format IQX_X, like IQ3_M. These are newer and offer better performance for their size.

These I-quants can also be used on CPU and Apple Metal, but will be slower than their K-quant equivalent, so speed vs performance is a tradeoff you'll have to decide.

The I-quants are *not* compatible with Vulcan, which is also AMD, so if you have an AMD card double check if you're using the rocBLAS build or the Vulcan build. At the time of writing this, LM Studio has a preview with ROCm support, and other inference engines have specific builds for ROCm.

</details>

## Credits

Thank you kalomaze and Dampf for assistance in creating the imatrix calibration dataset.

Thank you ZeroWw for the inspiration to experiment with embed/output.

Want to support my work? Visit my ko-fi page here: https://ko-fi.com/bartowski
