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

import abc
import insanity
import torch
import torch.nn as nn
import typing


class BaseEncoder(nn.Module, metaclass=abc.ABCMeta):
    """This is the base class that all encoders will extend."""

    @abc.abstractmethod
    def encode(
            self,
            input_seq: torch.Tensor,
            labels: torch.Tensor = None
    ) -> typing.Tuple[torch.FloatTensor, typing.Any]:
        """
        This computes the representations of the input tokens.
        Args:
            input_seq (:class::`torch.Tensor`): A tensor of vocabulary ids of tokens in the input sequence,
                (batch-size x max-seq-length).
            labels (:class::`torch.Tensor`): A tensor of labels for Masked Language Modelling.

        Returns:
            encodings (:class::`torch.FloatTensor`): A tensor of computed representations,
                (batch-size x max-seq-length x hidden-size).
            mlm_loss (:class::`torch.FloatTensor`): The masked language modelling loss when computed,
                otherwise None.
        """

    @property
    def encoder(self):
        """Specifies the encoder."""

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        """int: Specifies the dimension of token representations."""

    def forward(
            self,
            input_seq: torch.Tensor,
            labels: torch.Tensor = None
    ) -> typing.Tuple[torch.FloatTensor, typing.Any]:
        """
        This computes the representations of the input tokens.
        Args:
            input_seq (:class::`torch.Tensor`): A tensor of vocabulary ids of tokens in the input sequence,
                (batch-size x max-seq-length).
            labels (:class::`torch.Tensor`): A tensor of labels for Masked Language Modelling.

        Returns:
            encodings (:class::`torch.FloatTensor`): A tensor of computed representations,
                (batch-size x max-seq-length x hidden-size).
            mlm_loss (:class::`torch.FloatTensor`): The masked language modelling loss when computed,
                otherwise None.
        """

        # Sanitize args
        insanity.sanitize_type("input_seq", input_seq, torch.Tensor)
        if labels is not None:
            insanity.sanitize_type("labels", labels, torch.Tensor)
            assert input_seq.shape == labels.shape

        # Compute representations and return them.
        return self.encode(input_seq, labels)
