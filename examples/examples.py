# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2024 - Mtumbuka F.                                                    #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2024.1"
__date__ = "19 Dec, 2024."
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = ""
__status__ = "Development"

"""
This file demonstrates on how to use the pre-trained entity encoders.
"""

# Import the necessary libraries and classes

"""These are local classes in src/main/python/encore/pre_trained_enc"""
import encore.pre_trained_enc.pre_trained_albert_enc as albert_enc
import encore.pre_trained_enc.pre_trained_bert_enc as bert_enc
import encore.pre_trained_enc.pre_trained_roberta_enc as roberta_enc

"""Import tokenizers from the transformer library"""
from transformers import AutoTokenizer

"""To load the pre-trained encore model based on bert, and its associated tokenizer"""
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = bert_enc.EntityEncoder.from_pretrained("fmmka/bert-encore")

"""To load the pre-trained encore model based on albert, and its associated tokenizer"""
tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v1")
model = albert_enc.EntityEncoder.from_pretrained("fmmka/albert-encore")

"""To load the pre-trained encore model based on roberta, and its associated tokenizer"""
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = roberta_enc.EntityEncoder.from_pretrained("fmmka/roberta-encore")

"""Having loaded the tokenizer and pretrained model, below are the steps on how to encode text."""
# Example: Encode a sentence with an entity span
sentence = "The patient in front of her was waiting."
entity_position = [1]  # Index of "patient" in the tokenized sentence
inputs = tokenizer(sentence, return_tensors="pt")
# print(inputs)
# Compute embeddings
outputs, _ = model(input_ids=inputs["input_ids"])

entity_embedding = outputs[:, entity_position, :].squeeze(1)

