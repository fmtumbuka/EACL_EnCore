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

## Getting Started

### 1. Using the Pre-Trained Encoders

This section demonstrates how to use the pre-trained entity encoders.

#### Import the Necessary Libraries and Classes

```python
# Import local classes for pre-trained encoders from src/main/python/encore/pre_trained_enc
import encore.pre_trained_enc.pre_trained_albert_enc as albert_enc
import encore.pre_trained_enc.pre_trained_bert_enc as bert_enc
import encore.pre_trained_enc.pre_trained_roberta_enc as roberta_enc

# Import tokenizers from the transformers library
from transformers import AutoTokenizer

```
#### Load Pre-Trained Models and Tokenizers

Below are examples of how to load pre-trained encoders for BERT, ALBERT, and RoBERTa.
```python
# Load the pre-trained EnCore model based on BERT and its tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = bert_enc.EntityEncoder.from_pretrained("fmmka/bert-encore")

# Load the pre-trained EnCore model based on ALBERT and its tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v1")
model = albert_enc.EntityEncoder.from_pretrained("fmmka/albert-encore")

# Load the pre-trained EnCore model based on RoBERTa and its tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = roberta_enc.EntityEncoder.from_pretrained("fmmka/roberta-encore")
```

#### Encoding Text with Entity Spans
```python
# Example: Encode a sentence with an entity span
sentence = "The patient in front of her was waiting."
entity_position = [1]  # Index of the target word "patient" in the tokenized sentence

# Tokenize the input
inputs = tokenizer(sentence, return_tensors="pt")

# Compute embeddings using the pre-trained encoder
outputs, _ = model(input_ids=inputs["input_ids"])

# Extract the entity embedding for the specified entity position
entity_embedding = outputs[:, entity_position, :].squeeze(1)
```

Once the tokenizer and model are loaded, you can encode a sentence and extract entity embeddings for a specified entity span:

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


 
