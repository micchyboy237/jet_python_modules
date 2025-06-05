>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased')
>>> unmasker("Hello I'm a [MASK] model.")

[{'sequence': "[CLS] Hello I'm a model model. [SEP]",
  'score': 0.10182085633277893,
  'token': 13192,
  'token_str': 'model'},
 {'sequence': "[CLS] Hello I'm a world model. [SEP]",
  'score': 0.052126359194517136,
  'token': 11356,
  'token_str': 'world'},
 {'sequence': "[CLS] Hello I'm a data model. [SEP]",
  'score': 0.048930276185274124,
  'token': 11165,
  'token_str': 'data'},
 {'sequence': "[CLS] Hello I'm a flight model. [SEP]",
  'score': 0.02036019042134285,
  'token': 23578,
  'token_str': 'flight'},
 {'sequence': "[CLS] Hello I'm a business model. [SEP]",
  'score': 0.020079681649804115,
  'token': 14155,
  'token_str': 'business'}]