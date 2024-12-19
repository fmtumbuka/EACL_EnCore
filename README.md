# [EnCore: Fine-Grained Entity Typing by Pre-Training Entity Encoders on Coreference Chains](https://arxiv.org/abs/2305.12924) - EACL 2024.

EnCore is a state-of-the-art framework for fine-grained entity typing, leveraging pre-trained language models such as **BERT**, **ALBERT**, and **RoBERTa**. The encoders are trained with a novel contrastive learning approach on coreference chains to improve representation quality for co-referring entities. This repository provides tools for downloading and using the pre-trained encoders, as well as instructions on how to train an entity type classifier.

---

## **How It Works**

### **Pre-trained Encoder**
Our encoder refines existing pre-trained language models by applying **contrastive loss** to coreference chains:
- Co-referring entities are brought closer in the embedding space.
- Non-co-referring entities are pushed further apart.

### **Entity Type Classification**
We use the pre-trained encoder representations to classify entity types:
1. **Input:** A sentence with an entity span (e.g., *"the patient in front of her"*).
2. **Representation:** The encoder generates a representation for the head word of the entity span (*"patient"*).
3. **Prediction:** This representation is used to predict the entity type.

---

## **Getting Started**

### **1. Using the Pre-trained Encoders**

The pre-trained encoders are available on **HuggingFace**. Follow these steps to download and load them:

```python
from transformers import AutoTokenizer, AutoModel

# Choose an EnCore encoder: 'bert-encore', 'albert-encore', or 'roberta-encore'
encoder_name = "fmmka/bert-encore"
# For Albert use 'fmmka/albert-encore', for RoBerta use 'fmmka/roberta-encore'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(encoder_name)
model = AutoModel.from_pretrained(encoder_name)

# Example: Encode a sentence with an entity span
sentence = "The patient in front of her was waiting."
entity_position = [1]  # Index of "patient" in the tokenized sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Compute embeddings
outputs = model(**inputs)
entity_embedding = outputs.last_hidden_state[:, entity_position, :].squeeze(1)
print("Entity representation:", entity_embedding)
```

---

### **2. Training an Entity Type Classifier**

To train a classifier:

    1. Input Preparation: Use the encoder to generate representations for the head words of entity spans.
    2. Model Training: Train a simple feedforward neural network or any classifier on these representations using your labeled dataset.


### **How to Cite This Work**
```bibtex
@inproceedings{mtumbuka2024encore,
  title={EnCore: Fine-Grained Entity Typing by Pre-Training Entity Encoders on Coreference Chains},
  author={Mtumbuka, F. and Schockaert, S.},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1768--1781},
  year={2024},
  month={March}
}
```


 
