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
__date__ = "08 Mar 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import abc
import insanity
import torch
import torch.nn as nn
import typing


class BaseIntermediateLayer(nn.Module, metaclass=abc.ABCMeta):
    """This class is the base class for all intermediate operations between the encoder and classifier."""

    @abc.abstractmethod
    def forward(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.Tensor = None,
            cl_labels: torch.Tensor = None
    ) -> typing.Tuple[torch.FloatTensor, typing.Any]:
        """
        This computes the intermediate operation as specified in the experiment configuration.
        Args:
            input_seq (:class::`torch.FloatTensor`): A tensor of vector representations as computed by
                the specified encoder. (batch-size x seq-length x hidden-size).
            input_mask (:class::`torch.Tensor`): This mask indicates the position of co-referring named entities in the
                input sequence with `1`s and `0`s otherwise. (batch-size x seq-length).
            cl_labels (:class::`torch.Tensor`): These are contrastive learning labels that are used to compute loss.
                (1 x number of co-referring entities).

        Returns:
             representations (:class::`torch.FloatTensor`): A tensor of vectors from specific indices in
                the input tensor. Usually of batch-size * named entities x 1 x hidden-size.
            intermediate_loss (:class::`torch.Tensor`): The computed loss.
        """
        # Sanitize args.
        insanity.sanitize_type("input_seq", input_seq, torch.FloatTensor)
        if input_mask is not None:
            insanity.sanitize_type("input_mask", input_mask, torch.Tensor)
            assert (input_seq.shape[0], input_seq.shape[1]) == (input_mask.shape[0], input_mask.shape[1])

        if cl_labels is not None:
            insanity.sanitize_type("cl_labels", cl_labels, torch.Tensor)
            assert len(cl_labels.shape) == 1

    def retrieve_representations(
            self,
            input_seq: torch.FloatTensor,
            mask: torch.Tensor = None
    ) -> torch.FloatTensor:
        """
        This retrieves vectors at specific indices in a tensor.
        If the mask tensor is not specified, the method retrieves representations of the first
        element of each sequence, [CLS] token.
        Args:
            input_seq(:class::`torch.FloatTensor`): A tensor of vector representations as computed by
                the specified encoder. (batch-size x seq-length x hidden-size).
            mask (:class::`torch.Tensor`): This could either be a mask indicating the position of
                named entities in the input sequence, or masked out elements. (batch-size x seq-length)
        Returns:
            representations (:class::`torch.FloatTensor`): A tensor of vectors from specific indices in
                the input tensor. Usually of batch-size x 1 x hidden-size.
        """
        representations = None
        # Check that the sizes of the tensors are compatible.
        if mask is None:
            mask = torch.zeros(input_seq.shape[0], input_seq.shape[1])
            mask[:, 0] = 1
            if input_seq.is_cuda:
                mask = mask.cuda()

        # assert (input_seq.shape[0], input_seq.shape[1]) == (mask.shape[0], mask.shape[1])

        # Hidden-size
        hidden_size = input_seq.shape[2]

        # (batch-size, seq-len, hidden-size) -> (batch-size * seq-len, hidden-size)
        input_seq = input_seq.view(-1, hidden_size)

        # (batch-size, seq-len) -> (1, batch-size * seq-len)
        mask = mask.view(-1)

        # Find nonzero indices in the mask
        nonzero_indices = torch.nonzero(mask)

        # Retrieve specific vectors
        representations = input_seq[nonzero_indices.view(-1)]

        representations = representations.unsqueeze(1)

        return representations
