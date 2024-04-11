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
__date__ = "01 Apr 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import typing

import encore.classifier.base_classifier as base_classifier
import insanity
import torch
import torch.nn as nn


class MLPClassifier(base_classifier.BaseClassifier):
    """This implements an MLP based classifier."""

    def __init__(
            self,
            *args,
            entity_only: bool = False,
            hidden_layers: int = 1,
            dropout_rate: float = 0.00,
            **kwargs
    ):
        """
        This creates an instance of `MLPRelClassifier`.
        Args:
            *args: See meth:`base_relation_classifier.BaseRelationClassifier.__init__`
            entity_only (bool): This specifies whether to consider entities only or not.
            hidden_layers (int): The number of layers to use.
            dropout_rate (float): The dropout rate used for both attention and residual dropout.
            **kwargs: See meth:`base_relation_classifier.BaseRelationClassifier.__init__`
        """
        super().__init__(*args, **kwargs)

        # Sanitize args.
        insanity.sanitize_type("hidden_layers", hidden_layers, int)
        insanity.sanitize_type("dropout_rate", dropout_rate, float)
        insanity.sanitize_range("hidden_layers", hidden_layers, minimum=1)
        insanity.sanitize_range("dropout_rate", dropout_rate, minimum=0.00)

        # store args that are needed later on
        self._entity_only = entity_only
        self._hidden_layers = hidden_layers

        # create the MLP that is used to process input sequences
        layers = []
        last_size = self._input_size
        decay = (self._input_size - self._num_classes) // (self._hidden_layers + 1)
        for idx in range(self._hidden_layers):
            layers.append(nn.Linear(last_size, last_size - decay))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            last_size -= decay
        layers.append(nn.Linear(last_size, self._num_classes))
        self._mlp = nn.Sequential(*layers)

        self.reset_parameters()

    @property
    def hidden_layers(self) -> int:
        """int: Specifies the number of hidden layers."""
        return self._hidden_layers

    def compute_logits(self, input_seq: torch.FloatTensor, input_mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        This computes logits using the specified classifier.
        Args:
            input_seq (:class::`torch.FloatTensor`): The input sequence representations, (batch-size x seq-len x dim).
            input_mask (:class::`torch.LongTensor`): The mask with `1`s indicating the positions of interest, and `0`s
                otherwise. Mostly the positions of interest are where the entities are located.

        Returns:
            logits (:class::`torch.FloatTensor`): The logits tensor, (batch-size x seq-len x num-labels).
        """
        mask_indices = torch.nonzero(input_mask.view(-1))
        dim = input_seq.shape[2]
        input_seq = input_seq.view(-1, dim)[mask_indices.view(-1)]
        return self._mlp(input_seq)

    def _compute_logs(self, input_seq: torch.FloatTensor, input_mask: torch.LongTensor = None) -> torch.FloatTensor:
        logits = self.compute_logits(input_seq, input_mask)
        if self._entity_only:
            logits = logits * input_mask.unsqueeze(2)
        return logits

    def _compute_top(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.LongTensor = None,
            threshold: float = 0.5
    ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Compute logits and pass through the sigmoid function.
        logits = self.compute_logits(input_seq, input_mask).sigmoid()
        predictions = logits.clone().detach()
        # Find the maximum value: prediction
        max_index = torch.argmax(predictions, dim=1)

        # Set prediction true where there is the maximum value.
        for dim, i in enumerate(max_index):
            predictions[dim, i] = 1

        # For all logits above the threshold set as true predictions, otherwise false.
        predictions[predictions > threshold] = 1
        predictions[predictions != 1] = 0
        return predictions, logits

    def reset_parameters(self) -> None:
        """Resets all tunable parameters of the module."""
        for layer in self._mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

