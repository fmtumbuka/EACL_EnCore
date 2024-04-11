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
__date__ = "06 Mar 2023"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import datapipeline.cluster as cluster
import insanity
import typing


class Story(object):
    """This class encapsulates a single story."""

    def __init__(self, clusters: typing.List[cluster.Cluster], story_id: int):
        """
        This creates an instance of `Story`.
        Args:
            clusters (list): This is a list of clusters (co-reference chains) in the story.
            story_id (int): The id of the story as it is being read from the file.
        """

        # Sanitize args.
        insanity.sanitize_type("clusters", clusters, list)
        insanity.sanitize_type("story_id", story_id, int)
        insanity.sanitize_range("story_id", story_id, minimum=0)

        # Store args.
        self._clusters = clusters
        self._story_id = story_id

    def __len__(self) -> int:
        """int: This specifies the length of the story in terms of the number of clusters."""
        return len(self._clusters)

    @property
    def clusters(self) -> typing.List[cluster.Cluster]:
        """list: This returns clusters in the story."""
        return self._clusters

    @property
    def story_id(self) -> int:
        """int: This specifies the story id as read from the file."""
        return self._story_id
