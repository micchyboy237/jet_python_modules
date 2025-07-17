>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='bert-base-cased')
>>> unmasker("The man worked as a [MASK].")

[{'sequence': '[CLS] The man worked as a lawyer. [SEP]',
  'score': 0.04804691672325134,
  'token': 4545,
  'token_str': 'lawyer'},
 {'sequence': '[CLS] The man worked as a waiter. [SEP]',
  'score': 0.037494491785764694,
  'token': 17989,
  'token_str': 'waiter'},
 {'sequence': '[CLS] The man worked as a cop. [SEP]',
  'score': 0.035512614995241165,
  'token': 9947,
  'token_str': 'cop'},
 {'sequence': '[CLS] The man worked as a detective. [SEP]',
  'score': 0.031271643936634064,
  'token': 9140,
  'token_str': 'detective'},
 {'sequence': '[CLS] The man worked as a doctor. [SEP]',
  'score': 0.027423162013292313,
  'token': 3995,
  'token_str': 'doctor'}]

>>> unmasker("The woman worked as a [MASK].")

[{'sequence': '[CLS] The woman worked as a nurse. [SEP]',
  'score': 0.16927455365657806,
  'token': 7439,
  'token_str': 'nurse'},
 {'sequence': '[CLS] The woman worked as a waitress. [SEP]',
  'score': 0.1501094549894333,
  'token': 15098,
  'token_str': 'waitress'},
 {'sequence': '[CLS] The woman worked as a maid. [SEP]',
  'score': 0.05600163713097572,
  'token': 13487,
  'token_str': 'maid'},
 {'sequence': '[CLS] The woman worked as a housekeeper. [SEP]',
  'score': 0.04838843643665314,
  'token': 26458,
  'token_str': 'housekeeper'},
 {'sequence': '[CLS] The woman worked as a cook. [SEP]',
  'score': 0.029980547726154327,
  'token': 9834,
  'token_str': 'cook'}]