from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # `nomic-bert-2048` uses the standard BERT tokenizer

config = AutoConfig.from_pretrained('nomic-ai/nomic-bert-2048', trust_remote_code=True) # the config needs to be passed in
model = AutoModelForMaskedLM.from_pretrained('nomic-ai/nomic-bert-2048',config=config, trust_remote_code=True)

# To use this model directly for masked language modeling
classifier = pipeline('fill-mask', model=model, tokenizer=tokenizer,device="cpu")

print(classifier("I [MASK] to the store yesterday."))