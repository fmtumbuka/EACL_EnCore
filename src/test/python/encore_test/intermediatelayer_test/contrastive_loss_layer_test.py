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
__date__ = "09 Mar 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import encore.intermediatelayer.contrastive_loss_layer as contrastive_loss_layer
import torch
import torchtestcase


class ContrastiveLossLayerTest(torchtestcase.TorchTestCase):

    def test_init(self):
        with self.assertRaises(TypeError):
            contrastive_loss_layer.ContrastiveLossLayer(temperature=10)

        with self.assertRaises(TypeError):
            contrastive_loss_layer.ContrastiveLossLayer(temperature=-10)

        with self.assertRaises(ValueError):
            contrastive_loss_layer.ContrastiveLossLayer(temperature=-10.0)

    def test_retrieve_representations(self):
        cp_layer = contrastive_loss_layer.ContrastiveLossLayer()

        # Prepare input
        input_tensor = torch.FloatTensor(2, 5, 100)
        mask_tensor = torch.LongTensor(
            [
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0]
            ]
        )

        # get representations where the mask_tensor is 1
        retrieved_vectors = cp_layer.retrieve_representations(input_tensor, mask_tensor)

        # Check if the retrieved vectors are of the right shape and type.
        assert isinstance(retrieved_vectors, torch.FloatTensor)

        input_batch_size, _, input_dimension = input_tensor.shape
        r_batch_size, seq_len, r_dimension = retrieved_vectors.shape

        assert seq_len == 1  # The retrieved vectors have sequence length of 1
        assert (input_batch_size, input_dimension) == (r_batch_size, r_dimension)

    def test_forward(self):
        cp_loss = contrastive_loss_layer.ContrastiveLossLayer()
        embedding = torch.nn.Embedding(10, 3)
        input_ = torch.LongTensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8]
            ]
        )
        mask = torch.LongTensor(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ]
        )
        labels = torch.LongTensor(
            [
                [0],
                [1]
            ]
        )

        encoded_input = embedding(input_)
        representations, loss = cp_loss(encoded_input, mask=mask, labels=labels)

        # Check if the outputs are of correct types.
        assert isinstance(representations, torch.FloatTensor)
        assert isinstance(loss, torch.Tensor)

        # Check if the dimensions are as expected
        r_batch, r_seq_len, r_dim = representations.shape
        input_batch, _, input_dim = encoded_input.shape
        assert r_seq_len == 1
        assert (r_batch, r_seq_len, r_dim) == (input_batch, 1, input_dim)

        # Check if it raises type error
        with self.assertRaises(TypeError):
            cp_loss("encoded_input", mask=mask, labels=labels)


