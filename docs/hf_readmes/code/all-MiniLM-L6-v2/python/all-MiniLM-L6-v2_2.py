from transformers import AutoConfig, AutoModelForSequenceClassification
model_path = "nomic-ai/nomic-bert-2048"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# strict needs to be false here since we're initializing some new params
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, trust_remote_code=True, strict=False)