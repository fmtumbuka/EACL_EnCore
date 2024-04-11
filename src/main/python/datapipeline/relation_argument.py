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


class RelationArgument(object):
    """This class encapsulates arguments for relations."""
    def __init__(
            self,
            arg_role: str,
            arg_text: str,
            arg_text_tokens: list,
            arg_positions: list
    ):
        """
        This creates an instance of `RelationArgument`.
        Args:
            arg_role (str): The role the argument plays in a relation, either ARG-1 or ARG-2.
            arg_text (str): The argument text.
            arg_text_tokens (list): The list of tokens in argument text.
            arg_positions (list): The list containing the start and end positions of the argument in
                the input text, [start, end].
        """

        # Sanitize args.
        insanity.sanitize_type("arg_role", arg_role, str)
        insanity.sanitize_type("arg_text", arg_text, str)
        insanity.sanitize_type("arg_text_tokens", arg_text_tokens, list)
        insanity.sanitize_type("arg_positions", arg_positions, list)

        # Store args.
        self._arg_role = arg_role
        self._arg_text = arg_text
        self._arg_text_tokens = arg_text_tokens
        self._arg_positions = arg_positions

    @property
    def arg_role(self) -> str:
        """str: Specifies the role the argument plays in the relationship."""
        return self._arg_role

    @arg_role.setter
    def arg_role(self, arg_role: str) -> None:
        self._arg_role = str(arg_role)

    @property
    def arg_text(self) -> str:
        """str: Specifies the actual text for the argument."""
        return self._arg_text

    @arg_text.setter
    def arg_text(self, arg_text: str) -> None:
        self._arg_text = str(arg_text)

    @property
    def arg_text_tokens(self) -> typing.List:
        """list: Specifies a list of tokens in the argument text."""
        return self._arg_text_tokens

    @arg_text_tokens.setter
    def arg_text_tokens(self, arg_text_tokens: list) -> None:
        self._arg_text_tokens = arg_text_tokens

    @property
    def arg_positions(self) -> typing.List:
        """list: Specifies the positions of the argument text in the input text."""
        return self._arg_positions

    @arg_positions.setter
    def arg_positions(self, arg_positions: list) -> None:
        self._arg_positions = arg_positions
