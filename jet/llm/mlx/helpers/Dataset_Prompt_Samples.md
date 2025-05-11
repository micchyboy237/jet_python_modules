## Dataset Prompt Samples

```json
[
  {
    "category": "Classification",
    "structure": "Yes/No",
    "system_message": "Answer the question with 'Yes' or 'No' only.",
    "input": "Is Python a programming language?",
    "output": "Yes"
  },
  {
    "category": "Classification",
    "structure": "Text Classification",
    "system_message": "Classify the sentiment or category of the text as 'Positive', 'Negative', or another relevant label.",
    "input": "The service was terrible and the food was cold.",
    "output": "Negative"
  },
  {
    "category": "Classification",
    "structure": "Sentiment Analysis (Multi-Class)",
    "system_message": "Classify the sentiment of the text as 'Positive', 'Negative', 'Mixed', or another relevant category.",
    "input": "The movie was okay, but the ending felt rushed.",
    "output": "Mixed"
  },
  {
    "category": "Classification",
    "structure": "Fact-Checked Statements",
    "system_message": "Verify the statement and return 'True' or 'False' based on factual accuracy.",
    "input": "Vaccines cause autism.",
    "output": "False"
  },
  {
    "category": "Classification",
    "structure": "Text Entailment",
    "system_message": "Determine if the hypothesis is an 'Entailment', 'Contradiction', or 'Neutral' relative to the premise.",
    "input": {
      "premise": "All birds can fly.",
      "hypothesis": "A sparrow can fly."
    },
    "output": "Entailment"
  },
  {
    "category": "Selection",
    "structure": "Multiple Choice",
    "system_message": "Answer the following question by choosing one of the options provided without any additional text.\nOptions:\nMars\nEarth\nJupiter\nSaturn",
    "input": "Which planet is known as the Red Planet?",
    "output": "Mars"
  },
  {
    "category": "Selection",
    "structure": "Multiple Choice with Key",
    "system_message": "Answer the following question by choosing one of the option letters provided without any additional text.\nOptions:\nA) Oxygen\nB) Nitrogen\nC) Carbon Dioxide\nD) Argon",
    "input": "What is the most abundant gas in Earth's atmosphere?",
    "output": "B"
  },
  {
    "category": "Generation",
    "structure": "Text Completion",
    "system_message": "Complete the given text with the most appropriate word or phrase.",
    "input": "The largest ocean on Earth is the",
    "output": "Pacific Ocean"
  },
  {
    "category": "Generation",
    "structure": "Question-Answer Pairs",
    "system_message": "Provide a concise and accurate answer to the question.",
    "input": "Why do leaves change color in autumn?",
    "output": "Leaves change color in autumn due to the breakdown of chlorophyll, revealing pigments like carotenoids and anthocyanins."
  },
  {
    "category": "Generation",
    "structure": "Instruction-Response",
    "system_message": "Follow the instruction provided and generate a concise response.",
    "input": "Write a tagline for a coffee shop.",
    "output": "Brewed with love, served with warmth."
  },
  {
    "category": "Generation",
    "structure": "Dialogue",
    "system_message": "Continue the dialogue as the 'Bot' with a natural and relevant response.",
    "input": "User: Can you recommend a book? Bot:",
    "output": "Bot: Try '1984' by George Orwell for a thought-provoking read."
  },
  {
    "category": "Generation",
    "structure": "Generative Tasks",
    "system_message": "Generate creative content based on the provided prompt, ensuring coherence and relevance.",
    "input": "Write a short story about a lost astronaut.",
    "output": "Stranded on a crimson asteroid, astronaut Lila sent a final message to Earth, her voice steady despite the void. A faint signal blinked—hope, or just her suit's last spark?"
  },
  {
    "category": "Generation",
    "structure": "Text Generation (Conditional)",
    "system_message": "Generate text that meets the specified conditions or constraints in the prompt.",
    "input": "Generate a product description for a smartwatch with fitness tracking.",
    "output": "Track your fitness journey with our sleek smartwatch, featuring heart rate monitoring, step counting, and sleep analysis, all in a stylish, water-resistant design."
  },
  {
    "category": "Generation",
    "structure": "Question Generation",
    "system_message": "Generate a relevant question based on the provided statement or text.",
    "input": "The Eiffel Tower is 324 meters tall and was completed in 1889.",
    "output": "What is the height of the Eiffel Tower, and when was it completed?"
  },
  {
    "category": "Generation",
    "structure": "Code and Output",
    "system_message": "Execute the provided code and return its output.",
    "input": "x = [1, 2, 3]\nprint(sum(x))",
    "output": "6"
  },
  {
    "category": "Reasoning",
    "structure": "Reasoning Chains",
    "system_message": "Solve the problem step-by-step, providing a clear explanation and the final answer.",
    "input": "A store offers a 20%

 discount on a $50 item. What is the final price?",
    "output": "Calculate the discount: 20% of $50 = 0.20 × 50 = $10. Subtract from original price: $50 - $10 = $40. Final price is $40."
  },
  {
    "category": "Reasoning",
    "structure": "Commonsense Reasoning",
    "system_message": "Answer the question using commonsense knowledge, providing a clear explanation if needed.",
    "input": "If you leave a metal spoon in a hot pot, what happens?",
    "output": "The spoon becomes hot due to heat conduction from the pot."
  },
  {
    "category": "Extraction",
    "structure": "Named Entity Recognition (NER)",
    "system_message": "Identify and extract entities (e.g., Person, Product, Location) from the text, returning them in a structured format.",
    "input": "Elon Musk launched a Tesla Cybertruck in New York.",
    "output": {
      "entities": [
        {"type": "Person", "value": "Elon Musk"},
        {"type": "Product", "value": "Tesla Cybertruck"},
        {"type": "Location", "value": "New York"}
      ]
    }
  },
  {
    "category": "Extraction",
    "structure": "Document Context",
    "system_message": "Extract or summarize the requested information from the provided document context.",
    "input": "[Long research paper on AI ethics] Extract the main challenges.",
    "output": "Key challenges include bias in algorithms, lack of transparency, and ensuring equitable access to AI benefits."
  },
  {
    "category": "Extraction",
    "structure": "Relation Extraction",
    "system_message": "Extract the relationship between entities in the sentence, specifying the relation, subject, and object.",
    "input": "Barack Obama was born in Hawaii.",
    "output": {
      "relation": "born_in",
      "subject": "Barack Obama",
      "object": "Hawaii"
    }
  },
  {
    "category": "Extraction",
    "structure": "Keyword Extraction",
    "system_message": "Extract the key terms or phrases from the input text.",
    "input": "Artificial intelligence is transforming healthcare through diagnostics and personalized treatment.",
    "output": ["artificial intelligence", "healthcare", "diagnostics", "personalized treatment"]
  },
  {
    "category": "Transformation",
    "structure": "Translation Pairs",
    "system_message": "Translate the input text from the source language to the target language specified.",
    "input": "Thank you (English)",
    "output": "Merci (French)"
  },
  {
    "category": "Transformation",
    "structure": "Paraphrasing",
    "system_message": "Rewrite the input text to convey the same meaning using different words.",
    "input": "The meeting was postponed due to unforeseen circumstances.",
    "output": "The gathering was delayed because of unexpected issues."
  },
  {
    "category": "Transformation",
    "structure": "Text Summarization",
    "system_message": "Summarize the input text into a concise version, capturing the main points.",
    "input": "A 500-word article about the benefits of meditation, including stress reduction, improved focus, and better sleep.",
    "output": "Meditation reduces stress, enhances focus, and promotes better sleep, offering significant mental and physical health benefits."
  },
  {
    "category": "Transformation",
    "structure": "Text Correction",
    "system_message": "Correct any grammatical, spelling, or stylistic errors in the input text.",
    "input": "The dog run fastly to the park.",
    "output": "The dog runs quickly to the park."
  },
  {
    "category": "Transformation",
    "structure": "Text-to-SQL",
    "system_message": "Convert the natural language query into a valid SQL statement.",
    "input": "Find all employees with a salary greater than 50000 from the employee database.",
    "output": "SELECT * FROM employees WHERE salary > 50000;"
  },
  {
    "category": "Transformation",
    "structure": "Text-to-Regex",
    "system_message": "Convert the natural language request into a valid regular expression.",
    "input": "Extract all email addresses from a text.",
    "output": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
  },
  {
    "category": "Multimodal",
    "structure": "Multimodal Data",
    "system_message": "Describe the content of the provided image or multimodal input in detail.",
    "input": "[Image of a bustling city street at night] Describe the scene.",
    "output": "Neon lights illuminate a crowded city street, with taxis honking and pedestrians weaving through vibrant market stalls under a starry sky."
  },
  {
    "category": "Structural Analysis",
    "structure": "Text Segmentation",
    "system_message": "Split the input text into meaningful segments or sentences.",
    "input": "The conference starts at 9 AM and ends at 5 PM. Lunch is provided at noon.",
    "output": [
      "The conference starts at 9 AM and ends at 5 PM.",
      "Lunch is provided at noon."
    ]
  },
  {
    "category": "Structural Analysis",
    "structure": "Anaphora Resolution",
    "system_message": "Identify the anaphor (e.g., pronoun) and its referent in the text, returning them in a structured format.",
    "input": "Sarah forgot her keys. She had to call a locksmith.",
    "output": {
      "anaphor": "She",
      "referent": "Sarah"
    }
  }
]
```
