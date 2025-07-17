>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='roberta-large')
>>> unmasker("Hello I'm a <mask> model.")

[{'sequence': "<s>Hello I'm a male model.</s>",
  'score': 0.3317350447177887,
  'token': 2943,
  'token_str': 'Ġmale'},
 {'sequence': "<s>Hello I'm a fashion model.</s>",
  'score': 0.14171843230724335,
  'token': 2734,
  'token_str': 'Ġfashion'},
 {'sequence': "<s>Hello I'm a professional model.</s>",
  'score': 0.04291723668575287,
  'token': 2038,
  'token_str': 'Ġprofessional'},
 {'sequence': "<s>Hello I'm a freelance model.</s>",
  'score': 0.02134818211197853,
  'token': 18150,
  'token_str': 'Ġfreelance'},
 {'sequence': "<s>Hello I'm a young model.</s>",
  'score': 0.021098261699080467,
  'token': 664,
  'token_str': 'Ġyoung'}]