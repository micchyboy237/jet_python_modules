>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='roberta-large')
>>> unmasker("The man worked as a <mask>.")

[{'sequence': '<s>The man worked as a mechanic.</s>',
  'score': 0.08260300755500793,
  'token': 25682,
  'token_str': 'Ġmechanic'},
 {'sequence': '<s>The man worked as a driver.</s>',
  'score': 0.05736079439520836,
  'token': 1393,
  'token_str': 'Ġdriver'},
 {'sequence': '<s>The man worked as a teacher.</s>',
  'score': 0.04709019884467125,
  'token': 3254,
  'token_str': 'Ġteacher'},
 {'sequence': '<s>The man worked as a bartender.</s>',
  'score': 0.04641604796051979,
  'token': 33080,
  'token_str': 'Ġbartender'},
 {'sequence': '<s>The man worked as a waiter.</s>',
  'score': 0.04239227622747421,
  'token': 38233,
  'token_str': 'Ġwaiter'}]

>>> unmasker("The woman worked as a <mask>.")

[{'sequence': '<s>The woman worked as a nurse.</s>',
  'score': 0.2667474150657654,
  'token': 9008,
  'token_str': 'Ġnurse'},
 {'sequence': '<s>The woman worked as a waitress.</s>',
  'score': 0.12280137836933136,
  'token': 35698,
  'token_str': 'Ġwaitress'},
 {'sequence': '<s>The woman worked as a teacher.</s>',
  'score': 0.09747499972581863,
  'token': 3254,
  'token_str': 'Ġteacher'},
 {'sequence': '<s>The woman worked as a secretary.</s>',
  'score': 0.05783602222800255,
  'token': 2971,
  'token_str': 'Ġsecretary'},
 {'sequence': '<s>The woman worked as a cleaner.</s>',
  'score': 0.05576248839497566,
  'token': 16126,
  'token_str': 'Ġcleaner'}]