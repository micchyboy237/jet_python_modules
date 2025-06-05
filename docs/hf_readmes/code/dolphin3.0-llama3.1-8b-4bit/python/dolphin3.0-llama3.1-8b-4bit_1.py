>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-base-cased')
>>> unmasker("Hello I'm a [MASK] model.")

[{'sequence': "[CLS] Hello I'm a fashion model. [SEP]",
  'score': 0.09019174426794052,
  'token': 4633,
  'token_str': 'fashion'},
 {'sequence': "[CLS] Hello I'm a new model. [SEP]",
  'score': 0.06349995732307434,
  'token': 1207,
  'token_str': 'new'},
 {'sequence': "[CLS] Hello I'm a male model. [SEP]",
  'score': 0.06228214129805565,
  'token': 2581,
  'token_str': 'male'},
 {'sequence': "[CLS] Hello I'm a professional model. [SEP]",
  'score': 0.0441727414727211,
  'token': 1848,
  'token_str': 'professional'},
 {'sequence': "[CLS] Hello I'm a super model. [SEP]",
  'score': 0.03326151892542839,
  'token': 7688,
  'token_str': 'super'}]