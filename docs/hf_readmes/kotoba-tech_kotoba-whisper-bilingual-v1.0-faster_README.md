---
language: ja
tags:
  - audio
  - automatic-speech-recognition
license: mit
library_name: ctranslate2
---

# Whisper kotoba-whisper-bilingual-v1.0 model for CTranslate2

This repository contains the conversion of [kotoba-tech/kotoba-whisper-bilingual-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-bilingual-v1.0) 
to the [CTranslate2](https://github.com/OpenNMT/CTranslate2) model format.

This model can be used in CTranslate2 or projects based on CTranslate2 such as [faster-whisper](https://github.com/systran/faster-whisper).

## Example
Install library and download sample audio.
```shell
pip install faster-whisper
wget https://huggingface.co/datasets/japanese-asr/en_asr.esb_eval/resolve/main/sample.wav -O sample_en.wav
wget https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000/resolve/main/sample.flac -O sample_ja.flac
```

Inference with the kotoba-whisper-bilingual-v1.0-faster.

```python
from faster_whisper import WhisperModel

model = WhisperModel("kotoba-tech/kotoba-whisper-bilingual-v1.0-faster")

# Japanese ASR
segments, info = model.transcribe("sample_ja.flac", language="ja", task="transcribe", condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# English ASR
segments, info = model.transcribe("sample_en.wav", language="en", task="transcribe", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# Japanese (speech) to English (text) Translation
segments, info = model.transcribe("sample_ja.flac", language="en", task="translate", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# English (speech) to Japanese (text) Translation
segments, info = model.transcribe("sample_en.wav", language="ja", task="translate", chunk_length=15, condition_on_previous_text=False)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

```

### Benchmark
We measure the inference speed of different kotoba-whisper-v2.0 implementations with four different Japanese speech audio on MacBook Pro with the following spec:
- Apple M2 Pro
- 32GB
- 14-inch, 2023
- OS Sonoma Version 14.4.1 (23E224)

| audio file | audio duration (min)| [whisper.cpp](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml) (sec) | [faster-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) (sec)| [hf pipeline](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) (sec)
|--------|------|-----|------|-----|
|audio 1 | 50.3 | 581 | 2601 | 807 |
|audio 2 | 5.6  | 41  | 73   | 61  |
|audio 3 | 4.9  | 30  | 141  | 54  |
|audio 4 | 5.6  | 35  | 126  | 69  |

Scripts to re-run the experiment can be found bellow:
* [whisper.cpp](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0-ggml/blob/main/benchmark.sh)
* [faster-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0-faster/blob/main/benchmark.sh)
* [hf pipeline](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0/blob/main/benchmark.sh)

Also, currently whisper.cpp and faster-whisper support the [sequential long-form decoding](https://huggingface.co/distil-whisper/distil-large-v3#sequential-long-form),
and only Huggingface pipeline supports the [chunked long-form decoding](https://huggingface.co/distil-whisper/distil-large-v3#chunked-long-form), which we empirically
 found better than the sequnential long-form decoding.



## Conversion details

The original model was converted with the following command:

```
ct2-transformers-converter --model kotoba-tech/kotoba-whisper-bilingual-v1.0 --output_dir kotoba-whisper-bilingual-v1.0-faster --quantization float16
```

Note that the model weights are saved in FP16. This type can be changed when the model is loaded using the [`compute_type` option in CTranslate2](https://opennmt.net/CTranslate2/quantization.html).

## More information

For more information about the kotoba-whisper-v2.0, refer to the original [model card](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0).
