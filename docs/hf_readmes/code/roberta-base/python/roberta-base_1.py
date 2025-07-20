>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='roberta-base')
>>> unmasker("Hello I'm a <mask> model.")

[{'sequence': "<s>Hello I'm a male model.</s>",
  'score': 0.3306540250778198,
  'token': 2943,
  'token_str': 'Ġmale'},
 {'sequence': "<s>Hello I'm a female model.</s>",
  'score': 0.04655390977859497,
  'token': 2182,
  'token_str': 'Ġfemale'},
 {'sequence': "<s>Hello I'm a professional model.</s>",
  'score': 0.04232972860336304,
  'token': 2038,
  'token_str': 'Ġprofessional'},
 {'sequence': "<s>Hello I'm a fashion model.</s>",
  'score': 0.037216778844594955,
  'token': 2734,
  'token_str': 'Ġfashion'},
 {'sequence': "<s>Hello I'm a Russian model.</s>",
  'score': 0.03253649175167084,
  'token': 1083,
  'token_str': 'ĠRussian'}]