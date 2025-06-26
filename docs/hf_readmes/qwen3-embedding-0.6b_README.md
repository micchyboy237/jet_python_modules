---
license: apache-2.0
pipeline_tag: text-classification
language:
- en
widget:
- text: I ordered from you 2 weeks ago and its stil not here.
- text: I need to bring in my daughter for a checkup.
---
# Model Card: Fine-Tuned DistilBERT for User Intent Classification

## Model Description

The **Fine-Tuned DistilBERT** is a variant of the BERT transformer model,
distilled for efficient performance while maintaining high accuracy.
It has been adapted and fine-tuned for the specific task of classifying user intent in text data.

The model, named "distilbert-base-uncased," is pre-trained on a substantial amount of text data,
which allows it to capture semantic nuances and contextual information present in natural language text.
It has been fine-tuned with meticulous attention to hyperparameter settings, including batch size and learning rate, to ensure optimal model performance for the user intent classification task.

During the fine-tuning process, a batch size of 8 for efficient computation and learning was chosen.
Additionally, a learning rate (2e-5) was selected to strike a balance between rapid convergence and steady optimization,
ensuring the model not only learns quickly but also steadily refines its capabilities throughout training.

This model has been trained on a rather small dataset of under 50k, 100 epochs, specifically designed for user intent classification.
The dataset consists of text samples, each labeled with different user intents, such as "information seeking," "question asking," or "opinion expressing." The diversity within the dataset allowed the model to learn to identify user intent accurately. This dataset was carefully curated from a variety of sources.

The goal of this meticulous training process is to equip the model with the ability to classify user intent in text data effectively, making it ready to contribute to a wide range of applications involving user interaction analysis and personalization.

## Intended Uses & Limitations

### Intended Uses
- **User Intent Classification**: The primary intended use of this model is to classify user intent in text data. It is well-suited for applications that involve understanding user intentions, such as chatbots, virtual assistants, and recommendation systems.

### How to Use
To use this model for user intent classification, you can follow these steps:

```markdown
from transformers import pipeline

classifier = pipeline("text-classification", model="Falconsai/intent_classification")
text = "Your text to classify here."
result = classifier(text)
```

### Limitations
- **Specialized Task Fine-Tuning**: While the model excels at user intent classification, its performance may vary when applied to other natural language processing tasks. Users interested in employing this model for different tasks should explore fine-tuned versions available in the model hub for optimal results.

## Training Data

The model's training data includes a proprietary dataset designed for user intent classification. This dataset comprises a diverse collection of text samples, categorized into various user intent classes. The training process aimed to equip the model with the ability to classify user intent effectively.

### Training Stats
- Evaluation Loss:  0.011744413524866104
- Evaluation Accuracy: 0.9986976744186047
- Evaluation Runtime: 3.1136
- Evaluation Samples per Second: 1726.29
- Evaluation Steps per Second: 215.826

## Responsible Usage

It is essential to use this model responsibly and ethically, adhering to content guidelines and applicable regulations when implementing it in real-world applications, particularly those involving potentially sensitive content.

## References

- [Hugging Face Model Hub](https://huggingface.co/models)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

**Disclaimer:** The model's performance may be influenced by the quality and representativeness of the data it was fine-tuned on. Users are encouraged to assess the model's suitability for their specific applications and datasets.