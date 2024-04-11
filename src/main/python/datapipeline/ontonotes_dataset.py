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
__date__ = "24 May 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import insanity
import datapipeline.base_data_loader as base_data_loader
import datapipeline.ontonotes_example as ontonotes_example
import random
import torch.utils.data as data
import typing
import utilpackage.index_map as index_map


class OntoNotesDataset(data.Dataset):
    """This creates the OntoNotes dataset."""

    def __init__(
            self,
            data_loader: base_data_loader.BaseDataLoader,
            entity_labels: index_map.IndexMap
    ):
        """
        This creates an instance of `OntoNotesDataset`.
        Args:
            data_loader (::class:`base_data_loader.BaseDataLoader`): The OntoNotes data loader that reads data from the
                filesystem.
            entity_labels (::class:`index_map.IndexMap`): The list of entity labels mapped to indices.
        """
        super().__init__()

        # Sanitize args.
        insanity.sanitize_type("data_loader", data_loader, base_data_loader.BaseDataLoader)
        insanity.sanitize_type("entity_labels", entity_labels, index_map.IndexMap)

        # Store args.
        self._entity_labels = entity_labels
        self._data = data_loader.load()

    @property
    def entity_labels(self) -> index_map.IndexMap:
        return self._entity_labels

    def __getitem__(self, index) -> typing.List[ontonotes_example.OntoNotesExample]:
        """
        This returns an instance of `ontonotes_example.OntoNotesExample` at a given index.
        Args:
            index (int): The index for the example to retrieve.

        Returns:
            example (list[ontonotes_example.OntoNotesExample])
        """

        return [self._data[index]]

    def __len__(self) -> int:
        """int: Specifies the size of the dataset."""
        return len(self._data)

