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


class BaseClassifier(nn.Module, metaclass=abc.ABCMeta):
    """This is the base class that all classifiers will extend."""

    # Constructor
    def __init__(self, input_size: int, num_classes: int):
        """
        This creates an instance of `BaseRelationClassifier`.
        Args:
            input_size (int): This is the dimension of the input representations.
            num_classes (int): This is the total number of relation classes under consideration.
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("input_size", input_size, int)
        insanity.sanitize_type("num_classes", num_classes, int)
        insanity.sanitize_range("input_size", input_size, minimum=1)
        insanity.sanitize_range("num_classes", num_classes, minimum=1)

        # Store args.
        self._input_size = input_size
        self._num_classes = num_classes

    # Properties
    @property
    def input_size(self) -> int:
        """int: Specifies the dimension of the computed representations."""
        return self._input_size

    @property
    def num_classes(self) -> int:
        """int: Specifies the number of entity classes under consideration."""
        return self._num_classes

    @abc.abstractmethod
    def _compute_logs(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.LongTensor = None
    ) -> torch.FloatTensor:
        """
        This computes the probabilities over the label space.
        Args:
            input_seq (:class::`torch.FloatTensor`): The input sequence representations, (batch-size x seq-len x dim).
            input_mask (:class::`torch.LongTensor`): The mask with `1`s indicating the positions of interest, and `0`s
                otherwise. Mostly the positions of interest are where the entities are located.

        Returns:
            prob (:class::`torch.FloatTensor`): The probabilities over the label space.
        """

    @abc.abstractmethod
    def _compute_top(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.LongTensor = None,
            threshold: float = 0.5
    ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        This computes the top-k probabilities over the label space.
        Args:
            input_seq (:class::`torch.FloatTensor`): The input sequence representations, (batch-size x seq-len x dim).
            input_mask (:class::`torch.LongTensor`): The mask with `1`s indicating the positions of interest, and `0`s
                otherwise. Mostly the positions of interest are where the entities are located.
            threshold (float): The default threshold for the sigmoid predictions.

        Returns:
            top-k predictions: The top-k predicted label sequences.
        """

    def compute_logs(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.LongTensor = None
    ) -> torch.FloatTensor:
        """
        This computes the probabilities over the label space.
        Args:
            input_seq (:class::`torch.FloatTensor`): The input sequence representations, (batch-size x seq-len x dim).
            input_mask (:class::`torch.LongTensor`): The mask with `1`s indicating the positions of interest, and `0`s
                otherwise. Mostly the positions of interest are where the entities are located.

        Returns:
            prob (:class::`torch.FloatTensor`): The probabilities over the label space.
        """
        return self._compute_logs(input_seq, input_mask)

    def compute_top(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.LongTensor = None,
            threshold: float = 0.5
    ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        This computes the top-k probabilities over the label space.
        Args:
            input_seq (:class::`torch.FloatTensor`): The input sequence representations, (batch-size x seq-len x dim).
            input_mask (:class::`torch.LongTensor`): The mask with `1`s indicating the positions of interest, and `0`s
                otherwise. Mostly the positions of interest are where the entities are located.
            threshold
        Returns:
            top-k predictions: The top-k predicted label sequences.
        """
        return self._compute_top(input_seq, input_mask, threshold)

    def forward(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.LongTensor = None
    ) -> torch.FloatTensor:
        """
        This computes the entity types given sequence embeddings.
        Args:
            input_seq (:class::`torch.FloatTensor`): A sequence of input embeddings, a (batch-size x seq-len x dim)
                float tensor.
            input_mask (:class::`torch.FloatTensor`): A mask sequence with `1` indicating the position of entities
                and `0` indicating non-entity tokens.

        Returns:
            logits ()
        """
        return self.compute_logs(input_seq, input_mask)


