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
__date__ = "05 Apr 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import expbase as xb
import insanity
import torch
import typing


class EnCoreModelCheckpoint(xb.Checkpoint):
    """This represents a saved Entity Encoder Checkpoint."""

    def __init__(self, epoch: int, model_state: dict, optimizer_state: dict):
        """
        This creates an instance of the `EntityEncoderCheckpoint`.
        Args:
            epoch (int): The training epoch at which the encoder state was saved.
            model_state (dict): The state that describes the encoder at the end of epoch when the checkpoint was
                being saved.
            optimizer_state (dict): The state that describes the optimizer at the point when the checkpoint was being
                saved.
        """

        super().__init__()

        # Define and store attributes.
        self._epoch = epoch
        self._model_state = model_state
        self._optimizer_state = optimizer_state

    @property
    def epoch(self) -> int:
        """int: Specifies the training epoch after which the checkpoint was created."""
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def model_state(self) -> dict:
        """dict: Specifies the state of the encoder when the checkpoint was being created."""
        return self._model_state

    @model_state.setter
    def model_state(self, model_state: dict) -> None:
        self._model_state = model_state

    @property
    def optimizer_state(self) -> dict:
        """dict: Specifies the state of the optimizer when the checkpoint was being created."""
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, optimizer_state: dict) -> None:
        self._optimizer_state = optimizer_state

    def dump(self, path: str) -> typing.Any:
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "EnCoreModelCheckpoint":
        """
        This loads a checkpoint from a specific path.
        Args:
            path (str): The path of a checkpoint to load.

        Returns:
            Checkpoint: The loaded checkpoint.
        """
        return torch.load(path, map_location=torch.device("cpu"))
