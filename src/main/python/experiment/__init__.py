# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2022 - Mtumbuka F.                                                    #
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
__version__ = "2022.1"
__date__ = "28 Jul 2022"
__author__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import utilpackage.index_map as index_map

ACE_ENTITY_TYPES_COARSE_GRAINED = ['O', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

ACE_ENTITY_TYPES_COARSE_GRAINED_MAP = index_map.IndexMap(ACE_ENTITY_TYPES_COARSE_GRAINED)

ACE_ENTITY_TYPES_FINE_GRAINED = [
    'GPE:GPE-Cluster',
    'GPE:Nation',
    'GPE:Population-Center',
    'VEH:Underspecified',
    'WEA:Projectile',
    'FAC:Subarea-Facility',
    'WEA:Chemical',
    'LOC:Region-General',
    'ORG:Entertainment',
    'Contact-Info:URL',
    'WEA:Biological',
    'ORG:Medical-Science',
    'ORG:Educational',
    'Numeric:Percent',
    'LOC:Region-International',
    'WEA:Shooting',
    'GPE:State-or-Province',
    'VEH:Land',
    'Crime',
    'ORG:Government',
    'PER:Group',
    'Job-Title',
    'Contact-Info:E-Mail',
    'LOC:Celestial',
    'GPE:Special',
    'ORG:Media',
    'WEA:Nuclear',
    'Numeric:Money',
    'FAC:Plant',
    'WEA:Exploding',
    'LOC:Water-Body',
    'ORG:Sports',
    'FAC:Airport',
    'GPE:County-or-District',
    'PER:Indeterminate',
    'GPE:Continent',
    'FAC:Building-Grounds',
    'VEH:Subarea-Vehicle',
    'ORG:Religious',
    'ORG:Commercial',
    'FAC:Path',
    'LOC:Boundary',
    'Sentence',
    'WEA:Underspecified',
    'Contact-Info:Phone-Number',
    'PER:Individual',
    'VEH:Air',
    'ORG:Non-Governmental',
    'VEH:Water',
    'LOC:Land-Region-Natural',
    'WEA:Blunt',
    'WEA:Sharp',
    'LOC:Address',
    'O'
]
"""list: A list of golden entity types in ACE2005."""

ACE_ENTITY_TYPES_FINE_GRAINED_MAP = index_map.IndexMap(ACE_ENTITY_TYPES_FINE_GRAINED)
""":class::`index_map.IndexMap`: ACE entity types mapped to indices."""

ACE_RELATION_TYPES_FINE_GRAINED = [
    'PHYS:Near',
    'PER-SOC:Family',
    'ORG-AFF:Sports-Affiliation',
    'PER-SOC:Business',
    'ORG-AFF:Employment',
    'PART-WHOLE:Geographical',
    'GEN-AFF:Citizen-Resident-Religion-Ethnicity',
    'ORG-AFF:Membership',
    'ORG-AFF:Investor-Shareholder',
    'ORG-AFF:Ownership',
    'PART-WHOLE:Artifact',
    'PER-SOC:Lasting-Personal',
    'ORG-AFF:Founder',
    'ORG-AFF:Student-Alum',
    'PHYS:Located',
    'GEN-AFF:Org-Location',
    'ART:User-Owner-Inventor-Manufacturer',
    'PART-WHOLE:Subsidiary'
]
"""list: A list of fine-grained golden relation types in ACE2005."""

ACE_RELATION_TYPES = ['PER-SOC', 'ART', 'PHYS', 'PART-WHOLE', 'ORG-AFF', 'GEN-AFF']
"""list: A list of  golden relation types in ACE2005."""

ACE_RELATION_TYPES_MAP = index_map.IndexMap(ACE_RELATION_TYPES)

ACE_RELATION_TYPES_FINE_GRAINED_MAP = index_map.IndexMap(ACE_RELATION_TYPES_FINE_GRAINED)
""":class::`index_map.IndexMap`: ACE relation types mapped to indices."""

ALBERT_XX_LARGE_VERSION = "albert-xxlarge-v1"  # "albert-base-v2"
"""str: The version of ALBERT."""

BERT_BASE_CASED_VERSION = "bert-base-cased"
"""str: The cased version of BERT base"""

BERT_BASE_UNCASED_VERSION = "bert-base-uncased"
"""str: The  uncased version of BERT base."""

BEST_CKPT_FILE = "best.ckpt"
"""str: The filename of the stored checkpoint of the best model."""

DATASET_DIRS = {
    "ace_2005": "ace_2005",
    "figer": "figer",
    "gigaword": "gigaword",
    "ontonotes": "OntoNotes"

}
"""dict: This maps dataset names to their directories."""

FIGER_LABELS = [
    '/art',
    '/art/film',
    '/astral_body',
    '/award',
    '/biology',
    '/body_part',
    '/broadcast',
    '/broadcast/tv_channel',
    '/broadcast_network',
    '/broadcast_program',
    '/building',
    '/building/airport',
    '/building/dam',
    '/building/hospital',
    '/building/hotel',
    '/building/library',
    '/building/power_station',
    '/building/restaurant',
    '/building/sports_facility',
    '/building/theater',
    '/chemistry',
    '/computer',
    '/computer/algorithm',
    '/computer/programming_language',
    '/disease',
    '/education',
    '/education/department',
    '/education/educational_degree',
    '/event',
    '/event/attack',
    '/event/election',
    '/event/military_conflict',
    '/event/natural_disaster',
    '/event/protest',
    '/event/sports_event',
    '/event/terrorist_attack',
    '/finance',
    '/finance/currency',
    '/finance/stock_exchange',
    '/food',
    '/game',
    '/geography',
    '/geography/glacier',
    '/geography/island',
    '/geography/mountain',
    '/god',
    '/government',
    '/government/government',
    '/government/political_party',
    '/government_agency',
    '/internet',
    '/internet/website',
    '/language',
    '/law',
    '/living_thing',
    '/livingthing',
    '/livingthing/animal',
    '/location',
    '/location/body_of_water',
    '/location/bridge',
    '/location/cemetery',
    '/location/city',
    '/location/country',
    '/location/county',
    '/location/province',
    '/medicine',
    '/medicine/drug',
    '/medicine/medical_treatment',
    '/medicine/symptom',
    '/metropolitan_transit',
    '/metropolitan_transit/transit_line',
    '/military',
    '/music',
    '/news_agency',
    '/newspaper',
    '/organization',
    '/organization/airline',
    '/organization/company',
    '/organization/educational_institution',
    '/organization/fraternity_sorority',
    '/organization/sports_league',
    '/organization/sports_team',
    '/organization/terrorist_organization',
    '/park',
    '/people',
    '/people/ethnicity',
    '/person',
    '/person/actor',
    '/person/architect',
    '/person/artist',
    '/person/athlete',
    '/person/author',
    '/person/coach',
    '/person/director',
    '/person/doctor',
    '/person/engineer',
    '/person/monarch',
    '/person/musician',
    '/person/politician',
    '/person/religious_leader',
    '/person/soldier',
    '/person/terrorist',
    '/play',
    '/product',
    '/product/airplane',
    '/product/camera',
    '/product/car',
    '/product/computer',
    '/product/engine_device',
    '/product/instrument',
    '/product/mobile_phone',
    '/product/ship',
    '/product/spacecraft',
    '/product/weapon',
    '/rail',
    '/rail/railway',
    '/religion',
    '/religion/religion',
    '/software',
    '/time',
    '/title',
    '/train',
    '/transit',
    '/transportation',
    '/transportation/road',
    '/visual_art',
    '/visual_art/color',
    '/written_work'
]

FIGER_LABELS_MAP = index_map.IndexMap(FIGER_LABELS)

ONTONOTES_LABELS = [
    '/location', '/location/celestial', '/location/city', '/location/country', '/location/geography',
    '/location/geography/body_of_water', '/location/geography/island', '/location/geography/mountain',
    '/location/geograpy', '/location/geograpy/island', '/location/park', '/location/structure',
    '/location/structure/airport', '/location/structure/government', '/location/structure/hospital',
    '/location/structure/hotel', '/location/structure/restaurant', '/location/structure/sports_facility',
    '/location/structure/theater', '/location/transit', '/location/transit/bridge', '/location/transit/railway',
    '/location/transit/road', '/organization', '/organization/company', '/organization/company/broadcast',
    '/organization/company/news', '/organization/education', '/organization/government', '/organization/military',
    '/organization/music', '/organization/political_party', '/organization/sports_league', '/organization/sports_team',
    '/organization/stock_exchange', '/organization/transit', '/other', '/other/art', '/other/art/broadcast',
    '/other/art/film', '/other/art/music', '/other/art/stage', '/other/art/writing', '/other/award', '/other/body_part',
    '/other/currency', '/other/event', '/other/event/accident', '/other/event/election', '/other/event/holiday',
    '/other/event/natural_disaster', '/other/event/protest', '/other/event/sports_event',
    '/other/event/violent_conflict', '/other/food', '/other/health', '/other/health/malady', '/other/health/treatment',
    '/other/heritage', '/other/internet', '/other/language', '/other/language/programming_language', '/other/legal',
    '/other/living_thing', '/other/living_thing/animal', '/other/product', '/other/product/car',
    '/other/product/computer', '/other/product/mobile_phone', '/other/product/software', '/other/product/weapon',
    '/other/religion', '/other/scientific', '/other/sports_and_leisure', '/other/supernatural', '/person',
    '/person/artist', '/person/artist/actor', '/person/artist/author', '/person/artist/director',
    '/person/artist/music', '/person/athlete', '/person/coach', '/person/doctor', '/person/legal', '/person/military',
    '/person/political_figure', '/person/religious_leader', '/person/title'
]
"""list: The fine-grained OntoNotes 5 labels."""

ONTONOTES_LABELS_MAP = index_map.IndexMap(ONTONOTES_LABELS)

ROBERTA_LARGE_VERSION = "roberta-large"  # "roberta-base"
"""str: The version of RoBERTa"""
