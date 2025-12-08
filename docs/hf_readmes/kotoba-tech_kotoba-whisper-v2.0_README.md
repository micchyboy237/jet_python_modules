---
license: apache-2.0
language: ja
library_name: transformers
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
widget:
- example_title: CommonVoice 8.0 (Test Split)
  src: >-
    https://huggingface.co/datasets/japanese-asr/ja_asr.common_voice_8_0/resolve/main/sample.flac
- example_title: JSUT Basic 5000
  src: >-
    https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000/resolve/main/sample.flac
- example_title: ReazonSpeech (Test Split)
  src: >-
    https://huggingface.co/datasets/japanese-asr/ja_asr.reazonspeech_test/resolve/main/sample.flac
pipeline_tag: automatic-speech-recognition
metrics:
- wer
- cer
model-index:
- name: kotoba-tech/kotoba-whisper-v2.0
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: CommonVoice 8 (Japanese test set)
      type: japanese-asr/ja_asr.common_voice_8_0
    metrics:
    - name: WER
      type: WER
      value: 58.8
    - name: CER
      type: CER
      value: 9.2
  - task:
      type: automatic-speech-recognition
    dataset:
      name: ReazonSpeech (held out test set)
      type: japanese-asr/ja_asr.reazonspeech_test
    metrics:
    - name: WER
      type: WER
      value: 55.6
    - name: CER
      type: CER
      value: 11.6
  - task:
      type: automatic-speech-recognition
    dataset:
      name: JSUT Basic 5000
      type: japanese-asr/ja_asr.jsut_basic5000
    metrics:
    - name: WER
      type: WER
      value: 63.7
    - name: CER
      type: CER
      value: 8.4
datasets:
- japanese-asr/whisper_transcriptions.reazonspeech.all
- japanese-asr/whisper_transcriptions.reazonspeech.all.wer_10.0
- japanese-asr/whisper_transcriptions.reazonspeech.all.wer_10.0.vectorized
---

# Kotoba-Whisper (v2.0)
[**faster-whisper weight**](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster), [**whisper.cpp weight**](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml), [**pipeline with stable-ts/punctuation**](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.1)

_Kotoba-Whisper_ is a collection of distilled [Whisper](https://arxiv.org/abs/2212.04356) models for Japanese ASR, developed through the collaboration bewteen
[Asahi Ushio](https://asahiushio.com) and [Kotoba Technologies](https://twitter.com/kotoba_tech).
Following the original work of distil-whisper ([Robust Knowledge Distillation via Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430)), 
we employ OpenAI's [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) as the teacher model, and the student model consists the full encoder of the 
teacher large-v3 model and the decoder with two layers initialized from the first and last layer of the large-v3 model.
Kotoba-Whisper is **6.3x faster than large-v3**, while retaining as low error rate as the large-v3.

As successor of our first model, [kotoba-whisper-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0), we release ***kotoba-whisper-v2.0*** trained on the `all` subset of [ReazonSpeech](https://huggingface.co/datasets/reazon-research/reazonspeech) 
(the largest speech-transcription paired dataset in Japanese extracted from Japanese TV audio recordings), 
which amounts 7,203,957 audio clips (5 sec audio with 18 text tokens in average) after 
those transcriptions more than 10 WER are removed (see [WER Filter](https://huggingface.co/distil-whisper/distil-large-v3#wer-filter) for detail).
The model was trained for 8 epochs with batch size 256 with sampling rate of 16kHz, and the training and evaluation code to reproduce kotoba-whisper is available at [https://github.com/kotoba-tech/kotoba-whisper](https://github.com/kotoba-tech/kotoba-whisper).

Kotoba-whisper-v2.0 achieves better CER and WER than the [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) in the in-domain held-out test set
from ReazonSpeech, and achieves competitive CER and WER on the out-of-domain test sets including [JSUT basic 5000](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and
the Japanese subset from [CommonVoice 8.0](https://huggingface.co/datasets/common_voice) (see [Evaluation](#evaluation) for detail).


- ***CER***
| model                                                                                                                                             |   [CommonVoice 8 (Japanese test set)](https://huggingface.co/datasets/japanese-asr/ja_asr.common_voice_8_0) |   [JSUT Basic 5000](https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000) |   [ReazonSpeech (held out test set)](https://huggingface.co/datasets/japanese-asr/ja_asr.reazonspeech_test) |
|:--------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------:|------------------------------------------------------------------------------------------------------------:|
| [kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)                                                         |                                                                                                         9.2 |                                                                                     8.4 |                                                                                                        11.6 |
| [kotoba-tech/kotoba-whisper-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0)                                                         |                                                                                                         9.4 |                                                                                     8.5 |                                                                                                        12.2 |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)                                                                         |                                                                                                         8.5 |                                                                                     7.1 |                                                                                                        14.9 |
| [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)                                                                         |                                                                                                         9.7 |                                                                                     8.2 |                                                                                                        28.1 |
| [openai/whisper-large](https://huggingface.co/openai/whisper-large)                                                                               |                                                                                                        10   |                                                                                     8.9 |                                                                                                        34.1 |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium)                                                                             |                                                                                                        11.5 |                                                                                    10   |                                                                                                        33.2 |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base)                                                                                 |                                                                                                        28.6 |                                                                                    24.9 |                                                                                                        70.4 |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small)                                                                               |                                                                                                        15.1 |                                                                                    14.2 |                                                                                                        41.5 |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)                                                                                 |                                                                                                        53.7 |                                                                                    36.5 |                                                                                                       137.9 |



- ***WER***

| model                                                                                                                                             |   [CommonVoice 8 (Japanese test set)](https://huggingface.co/datasets/japanese-asr/ja_asr.common_voice_8_0) |   [JSUT Basic 5000](https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000) |   [ReazonSpeech (held out test set)](https://huggingface.co/datasets/japanese-asr/ja_asr.reazonspeech_test) |
|:--------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------:|------------------------------------------------------------------------------------------------------------:|
| [kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)                                                         |                                                                                                        58.8 |                                                                                    63.7 |                                                                                                        55.6 |
| [kotoba-tech/kotoba-whisper-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0)                                                         |                                                                                                        59.2 |                                                                                    64.3 |                                                                                                        56.4 |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)                                                                         |                                                                                                        55.1 |                                                                                    59.2 |                                                                                                        60.2 |
| [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)                                                                         |                                                                                                        59.3 |                                                                                    63.2 |                                                                                                        74.1 |
| [openai/whisper-large](https://huggingface.co/openai/whisper-large)                                                                               |                                                                                                        61.1 |                                                                                    66.4 |                                                                                                        74.9 |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium)                                                                             |                                                                                                        63.4 |                                                                                    69.5 |                                                                                                        76   |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base)                                                                                 |                                                                                                        87.2 |                                                                                    93   |                                                                                                        91.8 |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small)                                                                               |                                                                                                        74.2 |                                                                                    81.9 |                                                                                                        83   |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)                                                                                 |                                                                                                        93.8 |                                                                                    97.6 |                                                                                                        94.9 |


- ***Latency***: As kotoba-whisper uses the same architecture as [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3),
it inherits the benefit of the improved latency compared to [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) 
(**6.3x faster than large-v3**, see the table below taken from [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)).

| Model                                                                                        | Params / M | Rel. Latency |
|----------------------------------------------------------------------------------------------|------------|--------------|
| **[kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)**| **756**    | **6.3**      |
| **[kotoba-tech/kotoba-whisper-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0)**| **756**    | **6.3**      |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)                    | 1550       | 1.0          |


## Transformers Usage
Kotoba-Whisper is supported in the Hugging Face 🤗 Transformers library from version 4.39 onwards. To run the model, first 
install the latest version of Transformers. 

```bash
pip install --upgrade pip
pip install --upgrade transformers accelerate
```

### Short-Form Transcription
The model can be used with the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
class to transcribe short-form audio files (< 30-seconds) as follows:

```python
import torch
from transformers import pipeline
from datasets import load_dataset

# config
model_id = "kotoba-tech/kotoba-whisper-v2.0"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "ja", "task": "transcribe"}

# load model
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs
)

# load sample audio
dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
sample = dataset[0]["audio"]

# run inference
result = pipe(sample, generate_kwargs=generate_kwargs)
print(result["text"])
```

- To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline (make sure the audio is sampled in 16kHz):
```diff
- result = pipe(sample, generate_kwargs=generate_kwargs)
+ result = pipe("audio.mp3", generate_kwargs=generate_kwargs)
```

- For segment-level timestamps, pass the argument `return_timestamps=True` and return the `"chunks"` output:
```python
result = pipe(sample, return_timestamps=True, generate_kwargs=generate_kwargs)
print(result["chunks"])
```

***Sequential Long-Form:*** Kotoba-whisper is designed to be compatible with OpenAI's sequential long-form transcription algorithm. This algorithm uses a sliding window for buffered 
inference of long audio files (> 30-seconds), and returns more accurate transcriptions compared to the [chunked long-form algorithm](#chunked-long-form).
As default, if long audio files are passed to the model, it will transcribes with the sequential long-form transcription.
The sequential long-form algorithm should be used in either of the following scenarios:

1. Transcription accuracy is the most important factor, and latency is less of a consideration
2. You are transcribing **batches** of long audio files, in which case the latency of sequential is comparable to chunked, while being up to 0.5% WER more accurate

If you are transcribing single long audio files and latency is the most important factor, you should use the chunked algorithm
described [below](#chunked-long-form). For a detailed explanation of the different algorithms, refer to Sections 5 of 
the [Distil-Whisper paper](https://arxiv.org/pdf/2311.00430.pdf). The [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) 
class can be used to transcribe long audio files with the sequential algorithm as follows: 


### Chunked Long-Form
This algorithm should be used when a single large audio file is being transcribed and the fastest possible inference is required. In such circumstances, 
the chunked algorithm is up to 9x faster than OpenAI's sequential long-form implementation (see Table 7 of the [Distil-Whisper paper](https://arxiv.org/pdf/2311.00430.pdf)).
To enable chunking, pass the `chunk_length_s` parameter to the `pipeline`. For distil-large-v3, a chunk length of 25-seconds
is optimal. To activate batching over long audio files, pass the argument `batch_size`:

```python
import torch
from transformers import pipeline
from datasets import load_dataset

# config
model_id = "kotoba-tech/kotoba-whisper-v2.0"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "ja", "task": "transcribe"}

# load model
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs,
    batch_size=16
)

# load sample audio (concatenate instances to create a long audio)
dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
sample = {"array": np.concatenate([i["array"] for i in dataset[:20]["audio"]]), "sampling_rate": dataset[0]['audio']['sampling_rate']}

# run inference
result = pipe(sample, chunk_length_s=15, generate_kwargs=generate_kwargs)
print(result["text"])
```


### Additional Speed & Memory Improvements
You can apply additional speed and memory improvements to further reduce the inference speed and VRAM 
requirements. These optimisations primarily target the attention kernel, swapping it from an eager implementation to a 
more efficient flash attention version.

#### Flash Attention 2

We recommend using [Flash-Attention 2](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2) 
if your GPU allows for it. To do so, you first need to install [Flash Attention](https://github.com/Dao-AILab/flash-attention):

```
pip install flash-attn --no-build-isolation
```

Then pass `attn_implementation="flash_attention_2"` to `from_pretrained`:

```diff
- model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
+ model_kwargs = {"attn_implementation": "flash_attention_2"} if torch.cuda.is_available() else {}
```


## Model Details
See [https://huggingface.co/distil-whisper/distil-large-v3#model-details](https://huggingface.co/distil-whisper/distil-large-v3#model-details).


## Training
Please refer to [https://github.com/kotoba-tech/kotoba-whisper](https://github.com/kotoba-tech/kotoba-whisper) for the model training detail.
Datasets used in distillation and the whole model variations can be found at [https://huggingface.co/japanese-asr](https://huggingface.co/japanese-asr).


## Evaluation
The following code-snippets demonstrates how to evaluate the kotoba-whisper model on the Japanese subset of the CommonVoice 8.0. 
First, we need to install the required packages, including 🤗 Datasets to load the audio data, and 🤗 Evaluate to 
perform the WER calculation:

```bash
pip install --upgrade pip
pip install --upgrade transformers datasets[audio] evaluate jiwer
```

Evaluation can then be run end-to-end with the following example: 

```python
import torch
from transformers import pipeline
from datasets import load_dataset
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# model config
model_id = "kotoba-tech/kotoba-whisper-v2.0"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "japanese", "task": "transcribe"}
normalizer = BasicTextNormalizer()

# data config
dataset_name = "japanese-asr/ja_asr.reazonspeech_test"
audio_column = 'audio'
text_column = 'transcription'

# load model
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs,
    batch_size=16
)

# load the dataset and sample the audio with 16kHz
dataset = load_dataset(dataset_name, split="test")
transcriptions = pipe(dataset['audio'])
transcriptions = [normalizer(i['text']).replace(" ", "") for i in transcriptions]
references = [normalizer(i).replace(" ", "") for i in dataset['transcription']]

# compute the CER metric
cer_metric = load("cer")
cer = 100 * cer_metric.compute(predictions=transcriptions, references=references)
print(cer)
```

The huggingface links to the major Japanese ASR datasets for evaluation are summarized at [here](https://huggingface.co/collections/japanese-asr/japanese-asr-evaluation-dataset-66051a03d6ca494d40baaa26).
For example, to evaluate the model on JSUT Basic5000, change the `dataset_name`:

```diff
- dataset_name = "japanese-asr/ja_asr.reazonspeech_test"
+ dataset_name = "japanese-asr/ja_asr.jsut_basic5000"
```

## Acknowledgements
* [OpenAI](https://openai.com/) for the Whisper [model](https://huggingface.co/openai/whisper-large-v3).
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the model integration.
* Hugging Face 🤗 for the [Distil-Whisper codebase](https://github.com/huggingface/distil-whisper).
* [Reazon Human Interaction Lab](https://research.reazon.jp/) for the [ReazonSpeech dataset](https://huggingface.co/datasets/reazon-research/reazonspeech).