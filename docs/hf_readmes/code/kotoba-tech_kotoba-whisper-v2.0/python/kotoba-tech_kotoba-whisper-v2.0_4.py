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