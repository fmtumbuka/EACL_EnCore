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
__date__ = "23 May 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import insanity
import typing


class OntoNotesExample(object):
    """This class encapsulates an input example based on OntoNotes."""

    def __init__(
            self,
            input_tokens: typing.List[str],
            entity_start: int,
            entity_end: int,
            entity_head_pos: int,
            entity_labels: typing.List[str]
    ):
        """
        This creates an instance of `OntoNotesExample`.
        Args:
            input_tokens (list[str]): A list of words in the input sentence.
            entity_start (int): The start position of the entity span.
            entity_end (int): The end position of the entity span.
            entity_head_pos (int): The position of the syntactic head of the entity span.
            entity_labels (list[str]): The labels of the entity span.
        """

        # TODO:Sanitize args

        # Attributes
        self._entity_end = entity_end
        self._entity_head_pos = entity_head_pos
        self._entity_labels = entity_labels
        self._entity_start = entity_start
        self._input_tokens = input_tokens

    @property
    def entity_end(self) -> int:
        """"""
        return self._entity_end

    @entity_end.setter
    def entity_end(self, entity_end: int) -> None:
        self._entity_end = int(entity_end)

    @property
    def entity_head_pos(self) -> int:
        """"""
        return self._entity_head_pos

    @entity_head_pos.setter
    def entity_head_pos(self, entity_head_pos: int) -> None:
        self._entity_head_pos = entity_head_pos

    @property
    def entity_labels(self) -> typing.List[str]:
        """"""
        return self._entity_labels

    @entity_labels.setter
    def entity_labels(self, entity_labels: typing.List[str]) -> None:
        self._entity_labels = entity_labels

    @property
    def entity_start(self) -> int:
        """"""
        return self._entity_start

    @entity_start.setter
    def entity_start(self, entity_start) -> None:
        self._entity_start = entity_start

    @property
    def input_tokens(self) -> typing.List[str]:
        """"""
        return self._input_tokens

    @input_tokens.setter
    def input_tokens(self, input_tokens: typing.List[str]):
        self._input_tokens = input_tokens
