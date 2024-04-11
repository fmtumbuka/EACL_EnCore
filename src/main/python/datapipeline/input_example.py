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

import insanity
import typing


class InputExample(object):
    """This represents a single input example with its attributes."""

    # Constructor
    def __init__(
            self,
            ent_mask: typing.List[int],
            ent_labels: typing.List[str],
            input_tokens: typing.List[str],
    ):
        """
        This creates an instance of `InputExample.
        Args:
            ent_mask (list): The mask with `1`s indicating the positions of entities and `0`s non-entities.
            ent_labels (list): The list of tags for the entities.
            input_tokens (list): The actual input text split into tokens.
        """

        # Make sure the provided sequences are of the same length.
        if not len(ent_labels) == len(ent_mask) == len(input_tokens):
            raise ValueError(
                "The sequences are of different sizes; ent_labels vs ent_mask vs tokens: {} vs {} vs {}".format(
                    len(ent_labels), len(ent_mask), len(input_tokens)
                )
            )

        # Make sure the arguments are of the correct types.
        insanity.sanitize_type("ent_labels", ent_labels, list)
        insanity.sanitize_type("ent_mask", ent_mask, list)
        insanity.sanitize_type("input_tokens", input_tokens, list)

        # Create attributes
        self._ent_mask = ent_mask
        self._ent_labels = ent_labels
        self._input_tokens = input_tokens
        self._ent_labels_idx = None
        self._input_token_idx = None

    @property
    def ent_mask(self) -> typing.List[int]:
        """list[int]: Specifies the entity mask with `1`s indicating the positions of the entities, `0`s non-entities"""
        return self._ent_mask

    @ent_mask.setter
    def ent_mask(self, ent_mask: typing.List[int]) -> None:
        self._ent_mask = ent_mask

    @property
    def ent_labels(self) -> typing.List[str]:
        """list[str]: Specifies the list entity tags for the current example."""
        return self._ent_labels

    @ent_labels.setter
    def ent_labels(self, ent_labels: typing.List[str]) -> None:
        self._ent_labels = ent_labels

    @property
    def ent_labels_idx(self) -> typing.Union[None, typing.List[int]]:
        """(None or list[int]): Specifies the dictionary indices of the entity tags."""
        return self._ent_labels_idx

    @ent_labels_idx.setter
    def ent_labels_idx(self, ent_labels_idx: typing.List[int]) -> None:
        if not len(ent_labels_idx) == len(self._input_tokens):
            raise ValueError(
                "The sequence provided is of different length; input_tokens vs ent_labels_idx: {} vs {}".format(
                    len(self._input_tokens), len(ent_labels_idx)
                )
            )
        self._ent_labels_idx = ent_labels_idx

    @property
    def input_token_idx(self) -> typing.List[int]:
        """(None or list[int]): The sequence of vocabulary ids of the tokens in the input sequence."""
        return self._input_token_idx

    @input_token_idx.setter
    def input_token_idx(self, input_token_idx: typing.List[int]) -> None:
        self._input_token_idx = input_token_idx

    @property
    def input_tokens(self) -> typing.List[str]:
        """list[str]: Specifies the text tokens in the input sequence."""
        return self._input_tokens

    @input_tokens.setter
    def input_tokens(self, input_tokens: typing.List[str]) -> None:
        self._input_tokens = input_tokens
