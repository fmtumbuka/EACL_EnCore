# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2023 - Mtumbuka F.                                                       #
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
__version__ = "2023.1"
__date__ = "13 Mar 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import encore.classifier.base_classifier as base_classifier
import encore.encoder.base_encoder as base_encoder
import encore.intermediatelayer.base_intermediate_layer as base_intermediate_layer
import insanity
import torch
import torch.nn as nn
import typing
from transformers import AutoTokenizer


class EntityEncoder(nn.Module):
    """This encapsulates all modules for encoding as specified in the experiment configuration."""

    def __init__(
            self,
            classifier: base_classifier.BaseClassifier,
            encoder: base_encoder.BaseEncoder,
            intermediate_layer: base_intermediate_layer.BaseIntermediateLayer,
            tokenizer: AutoTokenizer
    ):
        """
        This creates an instance of `EntityEncoder`
        Args:
            classifier (:class::`base_classifier.BaseClassifier`): The classifier as specified in the experiment
                configuration.
            encoder (:class::`base_encoder.BaseEncoder`): The encoder as specified in the experiment configuration.
            intermediate_layer (:class::`base_intermediate_layer.BaseIntermediateLayer`): The intermediate layer as
                specified in the experiment configuration.
            tokenizer (:class::`transformers.AutoTokenizer`): The tokenizer being used based on the model selection.
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("classifier", classifier, base_classifier.BaseClassifier)
        insanity.sanitize_type("encoder", encoder, base_encoder.BaseEncoder)
        insanity.sanitize_type("intermediate_layer", intermediate_layer, base_intermediate_layer.BaseIntermediateLayer)

        # Create components
        self._classifier = classifier
        self._encoder = encoder
        self._intermediate_layer = intermediate_layer
        self._tokenizer = tokenizer

    @property
    def encoder(self):
        """Specifies the encoder being used."""
        return self._encoder

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Specifies the tokenizer being used."""
        return self._tokenizer

    def forward(
            self,
            input_seq: torch.Tensor,
            contrastive_labels: torch.Tensor = None,
            entity_mask: torch.Tensor = None,
            mlm_labels: torch.Tensor = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_seq (:class::`torch.LongTensor`): A tensor of vocabulary ids of tokens in the input sequence,
                (batch-size x max-seq-length).
            contrastive_labels (:class::`torch.LongTensor`): The labels to be used to compute loss during
                contrastive learning.
            entity_mask (:class::`torch.LongTensor`): This mask indicates the position of co-referring named entities in
                the input sequence, (batch-size x seq-length).
            mlm_labels (:class::`torch.LongTensor`): A tensor of labels for Masked Language Modelling.
                (batch-size x max-seq-length).
        Returns:
            mlm_loss (:class::`torch.Tensor`): The masked language modelling loss.
            cl_loss (:class::`torch.Tensor`): The contrastive learning loss.
        """

        # Sanitize args.
        insanity.sanitize_type("input_seq", input_seq, torch.Tensor)

        if contrastive_labels is not None:
            insanity.sanitize_type("contrastive_labels", contrastive_labels, torch.Tensor)
            assert len(contrastive_labels.shape) == 1

        if entity_mask is not None:
            insanity.sanitize_type("entity_mask", entity_mask, torch.Tensor)
            assert (input_seq.shape[0], input_seq.shape[1]) == (entity_mask.shape[0], entity_mask.shape[1])

        if mlm_labels is not None:
            insanity.sanitize_type("mlm_labels", mlm_labels, torch.Tensor)
            assert (input_seq.shape[0], input_seq.shape[1]) == (mlm_labels.shape[0], mlm_labels.shape[1])

        # Compute embeddings and mlm loss using the encoder.
        input_seq, mlm_loss = self._encoder(input_seq, mlm_labels)

        # Compute intermediate loss.
        _, intermediate_loss = self._intermediate_layer(input_seq, entity_mask, contrastive_labels)

        return mlm_loss, intermediate_loss
