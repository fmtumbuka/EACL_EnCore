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

import encore.encoder.bert_encoder as bert_encoder
import torch
import torchtestcase
import typing


class BertEncoderTest(torchtestcase.TorchTestCase):

    def test_init(self):
        # Check if wrong types raise typeError
        with self.assertRaises(TypeError):
            bert_encoder.BertEncoder(10)
        with self.assertRaises(TypeError):
            bert_encoder.BertEncoder("bert-base-uncased", 1.0)

        # Check if wrong values raise ValueError
        with self.assertRaises(ValueError):
            bert_encoder.BertEncoder("bert-base-uncased", -1)

        # Check if the encoder is an instance of BertMaskedLM
        encoder = bert_encoder.BertEncoder("bert-base-uncased")

    def test_encode(self):
        # First scenario with default bert settings
        encoder = bert_encoder.BertEncoder("bert-base-uncased")

        bert_input = torch.LongTensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10]
            ]
        )
        input_labels = bert_input

        # Test encode function
        output = encoder.encode(bert_input, input_labels)
        logits, _ = output
        assert isinstance(output, typing.Tuple)
        assert isinstance(logits, torch.FloatTensor)
        assert (logits.shape[0], logits.shape[1]) == (bert_input.shape[0], bert_input.shape[1])

        # First scenario with special tokens
        encoder = bert_encoder.BertEncoder("bert-base-uncased", num_tokens=6000)

        bert_input = torch.LongTensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10]
            ]
        )
        input_labels = bert_input

        # Test encode function
        output = encoder.encode(bert_input, input_labels)
        logits, _ = output
        assert isinstance(output, typing.Tuple)
        assert isinstance(logits, torch.FloatTensor)
        assert (logits.shape[0], logits.shape[1]) == (bert_input.shape[0], bert_input.shape[1])
        assert logits.shape[2] == 6000
