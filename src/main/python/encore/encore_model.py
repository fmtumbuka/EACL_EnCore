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
__date__ = "06 Feb 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import encore.classifier.base_classifier as base_classifier
import encore.encoder.base_encoder as base_encoder
import insanity
import torch
import torch.nn as nn
import typing
from transformers import AutoTokenizer


class EnCoreModel(nn.Module):
    """This model encapsulates all modules for the EnCoreModel as specified in the experiment configuration."""

    # Constructor.
    def __init__(
            self,
            classifier: base_classifier.BaseClassifier,
            encoder: base_encoder.BaseEncoder,
            tokenizer: AutoTokenizer
    ):
        """
        This creates a new instance of `EnCoreModel`.
        Args:
            classifier (:class::`base_classifier.BaseClassifier`): The classifier to use.
            encoder (:class::`base_encoder.BaseEncoder`): The encoder to use for entities.
            tokenizer ():
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("classifier", classifier, base_classifier.BaseClassifier)
        insanity.sanitize_type("encoder", encoder, base_encoder.BaseEncoder)

        # Store args.
        self._classifier = classifier
        self._encoder = encoder
        self._tokenizer = tokenizer

    # Properties.
    @property
    def classifier(self) -> base_classifier.BaseClassifier:
        """classifier (:class::`base_classifier.BaseClassifier`): Specifies the entity classifier being used."""
        return self._classifier

    @classifier.setter
    def classifier(self, classifier: base_classifier.BaseClassifier) -> None:
        insanity.sanitize_type("classifier", classifier, base_classifier.BaseClassifier)
        self._classifier = classifier

    @property
    def encoder(self) -> base_encoder.BaseEncoder:
        """encoder (:class::`base_encoder.BaseEncoder`): Specifies the encoder being used."""
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: base_encoder.BaseEncoder) -> None:
        insanity.sanitize_type("encoder", encoder, base_encoder.BaseEncoder)
        self._encoder = encoder

    @property
    def tokenizer(self) -> AutoTokenizer:
        """"""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: AutoTokenizer) -> None:
        self._tokenizer = tokenizer

    # Methods
    def compute_top(
            self,
            input_seq: torch.LongTensor,
            input_mask: torch.LongTensor = None,
            threshold: float = 0.5
    ) -> torch.FloatTensor:
        """
        This computes the top-k predictions.
        Args:
            input_seq (:class::`torch.LongTensor`): The input tensor, (batch-size x seq-len).
            input_mask (:class::`torch.LongTensor`, Optional): The mask indicating the positions of the entities.
            threshold (float): The probability threshold for considering a model prediction as true.
        Returns:
            top-k predictions () :
        """
        # Sanitize args.
        # if input_mask is not None:
        #     assert (input_mask.shape[0], input_mask.shape[1]) == (input_seq.shape[0], input_seq.shape[1])

        # Encode sequences
        input_seq, _ = self._encoder(input_seq, input_mask)
        return self._classifier.compute_top(input_seq, input_mask, threshold)

    def forward(
            self,
            input_seq: torch.LongTensor,
            input_mask: torch.LongTensor = None,
            mlm_labels: torch.Tensor = None
    ):
        """
        Args:
            input_seq (:class::`torch.LongTensor`): A tensor of vocabulary ids of tokens in the input sequence,
                (batch-size x max-seq-length).
            input_mask (:class::`torch.LongTensor`): This is a mask indicating the positions of named entities in the
                input sequence, (batch-size x seq-length).
            mlm_labels (:class::`torch.LongTensor`): A tensor of labels for Masked Language Modelling,
                (batch-size x max-seq-length).
        Returns:
            mlm_loss:
            predictions:
        """
        # # Sanitize args.
        # if input_mask is not None:
        #     assert (input_mask.shape[0], input_mask.shape[1]) == (input_seq.shape[0], input_seq.shape[1])
        #
        # if mlm_labels is not None:
        #     assert (mlm_labels.shape[0], mlm_labels.shape[1]) == (input_seq.shape[0], input_seq.shape[1])

        # Encode sequences, and compute masked language modelling loss.
        input_seq, mlm_loss = self._encoder(input_seq, mlm_labels)

        # Compute entity type predictions.
        predictions = self._classifier(input_seq, input_mask)

        return predictions, mlm_loss
