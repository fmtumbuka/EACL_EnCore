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

import collections.abc as abc
import typing
import insanity

T = typing.TypeVar('T')


class IndexMap(abc.Mapping, typing.Generic[T]):
    """Maps elements to integer indices."""

    def __init__(self, elements: typing.Sequence[T]):
        """Creates a new ``IndexMap`` for the provided sequence.

        Args:
            elements (sequence): The elements to create the ``IndexMap`` for, where indices in the ``sequence``
                correspond with indices in the ``IndexMap``.
        """
        # sanitize args
        insanity.sanitize_type("elements", elements, abc.Sequence)

        # store the provided elements
        self._all_values = list(elements)

        # create the mapping
        self._idx_to_value = list(elements)
        self._value_to_idx = {value: idx for idx, value in enumerate(self._idx_to_value)}

    #  MAGIC FUNCTIONS  ################################################################################################

    def __contains__(self, item: typing.Any):
        return item in self._value_to_idx

    def __getitem__(self, value: T) -> int:
        """Retrieves the index of the specified value."""
        return self._value_to_idx[value]

    def __iter__(self) -> typing.Iterator[typing.Tuple[int, T]]:
        return enumerate(self._idx_to_value)

    def __len__(self) -> int:
        return len(self._idx_to_value)

    #  METHODS  ########################################################################################################

    def all_values(self) -> typing.List[T]:
        """Retrieves a list of all values that indices are stored for.

        Returns:
            list: The list of values.
        """
        return list(self._all_values)

    def index(self, value: T) -> int:
        """Retrieves the index that is stored for the specified value.

        Args:
            value: The value of the index to retrieve.

        Returns:
            int: The index stored for ``value``.
        """
        return self._value_to_idx[value]

    def value(self, index: int) -> T:
        """Retrieves the value that is stored for the specified index.

        Args:
            index (int): The index of the element to retrieve.

        Returns:
            The element stored for ``index``.
        """
        return self._idx_to_value[index]
