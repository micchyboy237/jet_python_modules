>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
>>> unmasker("Hello I'm a [MASK] model.")

[{'sequence': "[CLS] hello i'm a role model. [SEP]",
  'score': 0.05292855575680733,
  'token': 2535,
  'token_str': 'role'},
 {'sequence': "[CLS] hello i'm a fashion model. [SEP]",
  'score': 0.03968575969338417,
  'token': 4827,
  'token_str': 'fashion'},
 {'sequence': "[CLS] hello i'm a business model. [SEP]",
  'score': 0.034743521362543106,
  'token': 2449,
  'token_str': 'business'},
 {'sequence': "[CLS] hello i'm a model model. [SEP]",
  'score': 0.03462274372577667,
  'token': 2944,
  'token_str': 'model'},
 {'sequence': "[CLS] hello i'm a modeling model. [SEP]",
  'score': 0.018145186826586723,
  'token': 11643,
  'token_str': 'modeling'}]