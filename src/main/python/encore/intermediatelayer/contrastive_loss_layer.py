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

import encore.intermediatelayer.base_intermediate_layer as base_intermediate_layer
import insanity
import torch
import typing
from pytorch_metric_learning import losses


class ContrastiveLossLayer(base_intermediate_layer.BaseIntermediateLayer):
    """This computes the contrastive loss."""

    def __init__(self, temperature: float = 0.07):
        """
        This creates an instance of the `ContrastiveLossLayer`.
        Args:
            temperature (float): This is tau in the contrastive loss equation.
        """

        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("temperature", temperature, float)
        insanity.sanitize_range("temperature", temperature, minimum=0.00)

        # Create loss function.
        self._contrastive_loss = losses.NTXentLoss(temperature=temperature)

    def forward(
            self,
            input_seq: torch.FloatTensor,
            input_mask: torch.Tensor = None,
            cl_labels: torch.Tensor = None
    ) -> typing.Tuple[torch.FloatTensor, typing.Any]:
        # Args already sanitized in the base class, see base_intermediate_later.BaseIntermediateLayer.forward...

        # Extract entity representations.
        entity_representations = self.retrieve_representations(input_seq, input_mask)

        # Compute contrastive loss.
        cp_loss = 0
        try:
            cp_loss = self._contrastive_loss(entity_representations.squeeze(1), cl_labels)
        except Exception as e:
            pass

        return entity_representations, cp_loss

