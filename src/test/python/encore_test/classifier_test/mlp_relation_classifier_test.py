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
__date__ = "02 Apr 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import encore.classifier.mlp_classifier as mlp_classifier
import torch
import torchtestcase


class TestMLPClassifier(torchtestcase.TorchTestCase):
    """This implements the test cases for the `mlp_classifier.MLPClassifier`"""
    def test_init(self):
        dropout_rate = 0.1
        hidden_layers = 3
        input_size = 20
        num_classes = 5

        with self.assertRaises(TypeError):
            # Checks if a TypeError is raised when wrong type is given for input_size.
            mlp_classifier.MLPClassifier(
                input_size=dropout_rate,
                num_classes=num_classes,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            )

        with self.assertRaises(TypeError):
            # Checks if a TypeError is raised when wrong type is given for num_classes.
            mlp_classifier.MLPClassifier(
                input_size=input_size,
                num_classes=dropout_rate,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            )

        with self.assertRaises(TypeError):
            # Checks if a TypeError is raised when wrong type is given for hidden_layers.
            mlp_classifier.MLPClassifier(
                input_size=input_size,
                num_classes=num_classes,
                hidden_layers=dropout_rate,
                dropout_rate=dropout_rate
            )

        with self.assertRaises(TypeError):
            # Checks if a TypeError is raised when wrong type is given for dropout_rate.
            mlp_classifier.MLPClassifier(
                input_size=input_size,
                num_classes=num_classes,
                hidden_layers=hidden_layers,
                dropout_rate=input_size
            )

        with self.assertRaises(ValueError):
            # Checks if a ValueError is raised when wrong value is given for input_size.
            mlp_classifier.MLPClassifier(
                input_size=-input_size,
                num_classes=num_classes,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            )
        with self.assertRaises(ValueError):
            # Checks if a ValueError is raised when wrong value is given for num_classes.
            mlp_classifier.MLPClassifier(
                input_size=input_size,
                num_classes=-num_classes,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            )

        with self.assertRaises(ValueError):
            # Checks if a ValueError is raised when wrong value is given for hidden_layers.
            mlp_classifier.MLPClassifier(
                input_size=input_size,
                num_classes=num_classes,
                hidden_layers=-hidden_layers,
                dropout_rate=dropout_rate
            )

        with self.assertRaises(ValueError):
            # Checks if a ValueError is raised when wrong value is given for dropout_rate.
            mlp_classifier.MLPClassifier(
                input_size=input_size,
                num_classes=num_classes,
                hidden_layers=hidden_layers,
                dropout_rate=-dropout_rate
            )

        classifier = mlp_classifier.MLPClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        # Check if the created instance has the correct values for input_size, hidden_layers, num_layers
        assert input_size == classifier.input_size
        assert hidden_layers == classifier.hidden_layers
        assert num_classes == classifier.num_classes

    def test_forward(self):
        dropout_rate = 0.1
        hidden_layers = 3
        input_size = 20
        num_classes = 5

        input_tensor = torch.FloatTensor(3, 30, input_size)

        # Checks if the forward function operates as intended.
        classifier = mlp_classifier.MLPClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        logits = classifier(input_tensor)

        # Checks if the output of the classifier is a FloatTensor
        assert isinstance(logits, torch.FloatTensor)

        # Checks if the dimensions of the logits are as expected: (batch-size x seq-len x num-classes)
        assert (logits.shape[0], logits.shape[1], logits.shape[2]) == (
            input_tensor.shape[0], input_tensor.shape[1], num_classes
        )
