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

import datapipeline.ace_entity_data_loader as ace_entity_data_loader
import datapipeline.ace_entity_dataset as ace_entity_dataset
import datapipeline.cluster as cluster
import datapipeline.figer_loader as figer_loader
import datapipeline.giga_word_dataset as giga_word_dataset
import datapipeline.giga_word_loader as giga_word_loader
import datapipeline.ontonotes_dataset as ontonotes_dataset
import datapipeline.ontonotes_example as ontonotes_example
import datapipeline.ontonotes_loader as ontonotes_loader
import encore.encore_model as encore_model
import encore.entity_encoder as entity_encoder
import encore.classifier.base_classifier as base_classifier
import encore.classifier.mlp_classifier as mlp_classifier
import encore.classifier.no_classifier as no_classifier
import encore.encoder.albert_encoder as albert_encoder
import encore.encoder.base_encoder as base_encoder
import encore.encoder.bert_encoder as bert_encoder
import encore.encoder.roberta_encoder as roberta_encoder
import encore.intermediatelayer.base_intermediate_layer as base_intermediate_layer
import encore.intermediatelayer.contrastive_loss_layer as contrastive_loss_layer
import encore.intermediatelayer.no_intermediate_layer as no_intermediate_layer
import expbase.util as util
import experiment
import experiment.config as config
import insanity
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import typing
import utilpackage.index_map as index_map
from transformers import AutoTokenizer

import numpy as np


class ComponentFactory(object):
    """Creates components from the user defined configuration for an experiment."""

    @classmethod
    def _create_classifier(cls, conf: config.Config, enc_hidden_size: int) -> base_classifier.BaseClassifier:
        """
        This creates a classifier based on the specified experiment configurations.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
            enc_hidden_size (int): The hidden size of the encoder.
        Returns:
            classifier (::class:`base_classifier.BaseClassifier`): The new classifier.
        """
        num_classes = conf.num_classes
        if conf.ace_dataset:
            num_classes = len(experiment.ACE_ENTITY_TYPES_COARSE_GRAINED_MAP)
        if conf.ontonotes:
            num_classes = len(experiment.ONTONOTES_LABELS_MAP)
        if conf.figer:
            num_classes = len(experiment.FIGER_LABELS_MAP)
        classifier = no_classifier.NoClassifier(input_size=enc_hidden_size, num_classes=num_classes)
        if not conf.pretrain:
            if conf.mlp_classifier:
                classifier = mlp_classifier.MLPClassifier(
                    input_size=enc_hidden_size,
                    num_classes=num_classes,
                    entity_only=conf.entity_only,
                    hidden_layers=conf.classifier_layers,
                    dropout_rate=conf.dropout_rate
                )

        return classifier

    @classmethod
    def _create_entity_encoder(cls, conf: config.Config) -> base_encoder.BaseEncoder:
        """
        This creates an entity encoder based on the specified experiment configuration.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
        Returns:
            encoder (::class:`base_encoder.BaseEncoder`): The new encoder.
        """

        # The default entity encoder is the BERT encoder.
        encoder = bert_encoder.BertEncoder(experiment.BERT_BASE_UNCASED_VERSION)

        if conf.albert_enc_ent:
            # Create ALBERT based entity encoder.
            encoder = albert_encoder.AlbertEncoder(experiment.ALBERT_XX_LARGE_VERSION)

        if conf.bert_enc_ent_cased:
            encoder = bert_encoder.BertEncoder(experiment.BERT_BASE_CASED_VERSION)

        if conf.roberta_enc_ent:
            # Create RoBERTa based entity encoder.
            encoder = roberta_encoder.RoBERTaEncoder(experiment.ROBERTA_LARGE_VERSION)

        # Freeze encoder if specified in the experiment configuration.
        if conf.freeze_ent_enc:
            for p in encoder.parameters():
                p.requires_grad = False

        return encoder

    @classmethod
    def _create_intermediate_layer(cls, conf: config.Config) -> base_intermediate_layer.BaseIntermediateLayer:
        """
        This creates an intermediate layer based on the specified experiment configuration.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
        Returns:
            intermediate_layer (::class:`base_intermediate_layer.BaseIntermediateLayer`): The new intermediate layer.
        """

        # The default intermediate layer is no intermediate layer at all.
        intermediate_layer = no_intermediate_layer.NoIntermediateLayer()

        if conf.cp_layer:
            intermediate_layer = contrastive_loss_layer.ContrastiveLossLayer(conf.cp_loss_tau)

        return intermediate_layer

    @classmethod
    def create_entity_encoder(cls, conf: config.Config) -> base_encoder.BaseEncoder:
        """
        This creates an entity encoder based on the specified experiment configuration.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
        Returns:
            encoder (::class:`base_encoder.BaseEncoder`): The new encoder.
        """
        return cls._create_entity_encoder(conf)

    @classmethod
    def create_dataset(cls, conf: config.Config, dev: bool = False, test: bool = False) -> data.Dataset:
        """

        Args:
            conf:
            dev:
            test:
        Returns:

        """
        dataset = None
        pickle_file = None

        # Default tokenizer is for the BERT model
        tokenizer_name = experiment.BERT_BASE_UNCASED_VERSION
        if conf.albert_enc_ent:
            tokenizer_name = experiment.ALBERT_XX_LARGE_VERSION
        if conf.roberta_enc_ent:
            tokenizer_name = experiment.ROBERTA_LARGE_VERSION
        if conf.bert_enc_ent_cased:
            tokenizer_name = experiment.BERT_BASE_CASED_VERSION

        if conf.ace_dataset:
            ace_data_path = os.path.join(conf.data_dir, "ace_2005")
            data_path = ace_data_path
            if dev:
                data_path = os.path.join(data_path, "dev")
            elif test:
                data_path = os.path.join(data_path, "test")
            else:
                data_path = os.path.join(data_path, "train")

            if not conf.ace_fine_grained:
                data_path = f"{data_path}.general"
            else:
                data_path = f"{data_path}.fine-grained"

            if conf.rel_mask:
                data_path = f"{data_path}.mask"

            pickle_file = f"{data_path}.{tokenizer_name}.ent.data.pickle"
            if os.path.isfile(pickle_file):
                print("Loading data from {}.".format(pickle_file))
                dataset = util.better_pickle.pickle_load(pickle_file)
            else:
                # Default tokenizer is for the BERT model
                tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_VERSION)

                if conf.albert_enc_ent:
                    tokenizer = AutoTokenizer.from_pretrained(experiment.ALBERT_XX_LARGE_VERSION)

                if conf.roberta_enc_ent:
                    tokenizer = AutoTokenizer.from_pretrained(experiment.ROBERTA_LARGE_VERSION)
                print("Loading data from {}...".format(data_path))
                data_loader = ace_entity_data_loader.ACEEntityDataLoader(
                    ace_data_path,
                    dev=dev,
                    test=test,
                    fine_grained=conf.ace_fine_grained
                )
                entity_types = None
                if conf.ace_fine_grained:
                    entity_types = experiment.ACE_ENTITY_TYPES_FINE_GRAINED_MAP
                else:
                    entity_types = experiment.ACE_ENTITY_TYPES_COARSE_GRAINED_MAP
                dataset = ace_entity_dataset.ACEEntityDataset(
                    data_loader=data_loader,
                    tokenizer=tokenizer,
                    entity_types=entity_types
                )
                print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                util.better_pickle.pickle_dump(dataset, pickle_file)
                print("OK")
        elif conf.ontonotes:
            data_path = os.path.join(conf.data_dir, experiment.DATASET_DIRS["ontonotes"])
            partition = None
            if dev:
                partition = "dev/"
            elif test:
                partition = "test/"
            else:
                partition = "train/"
            pickle_file = None
            pickle_file_path = os.path.join(data_path, partition)
            if test and conf.order_one:
                pickle_file = f"{pickle_file_path}order_one.data.pickle"

            elif test and conf.order_two:
                pickle_file = f"{pickle_file_path}order_two.data.pickle"

            elif test and conf.order_three:
                pickle_file = f"{pickle_file_path}order_three.data.pickle"

            else:
                pickle_file = f"{pickle_file_path}data.pickle"

            if os.path.isfile(pickle_file):
                print("Loading data from {}.".format(pickle_file))
                dataset = util.better_pickle.pickle_load(pickle_file)
            else:
                data_loader = ontonotes_loader.OntoNotesLoader(
                    data_path,
                    dev=dev,
                    test=test,
                    order_one=conf.order_one,
                    order_two=conf.order_two,
                    order_three=conf.order_three
                )
                dataset = ontonotes_dataset.OntoNotesDataset(data_loader, experiment.ONTONOTES_LABELS_MAP)
                print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                util.better_pickle.pickle_dump(dataset, pickle_file)
                print("OK")
        elif conf.figer:
            data_path = os.path.join(conf.data_dir, experiment.DATASET_DIRS["figer"])
            partition = None
            if dev:
                partition = "dev/"
            elif test:
                partition = "test/"
            else:
                partition = "train/"
            pickle_file = None
            pickle_file_path = os.path.join(data_path, partition)
            if test and conf.order_one:
                pickle_file = f"{pickle_file_path}order_one.data.pickle"

            elif test and conf.order_two:
                pickle_file = f"{pickle_file_path}order_two.data.pickle"

            elif test and conf.order_three:
                pickle_file = f"{pickle_file_path}order_three.data.pickle"

            else:
                pickle_file = f"{pickle_file_path}data.pickle"

            if os.path.isfile(pickle_file):
                print("Loading data from {}.".format(pickle_file))
                dataset = util.better_pickle.pickle_load(pickle_file)
            else:
                data_loader = figer_loader.FigerLoader(
                    data_path,
                    dev=dev,
                    test=test,
                    order_one=conf.order_one,
                    order_two=conf.order_two,
                    order_three=conf.order_three
                )
                dataset = ontonotes_dataset.OntoNotesDataset(data_loader, experiment.FIGER_LABELS_MAP)
                print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                util.better_pickle.pickle_dump(dataset, pickle_file)
                print("OK")
        else:
            if conf.pretrain:
                gigaword_data_path = os.path.join(conf.data_dir, "gigaword/")
                data_path = gigaword_data_path
                if conf.merge_datasets:
                    files = []
                    for r, d, f in os.walk(data_path):
                        for file in f:
                            if file.endswith(".pickle"):
                                files.append(os.path.join(r, file))
                    datasets = []
                    for file_name in files:
                        print("Loading data from {} ...".format(file_name))
                        datasets.append(util.better_pickle.pickle_load(file_name))
                        print("OK")
                    print()
                    print("Merging datasets...")
                    dataset = data.ConcatDataset(datasets)
                    print("Ok")
                    print()
                else:
                    # pickle_file = f"{data_path}gigaword.{tokenizer_name}.data-2.pickle"
                    pickle_file = f"{data_path}gigaword.bert-base-uncased.data-8099.pickle"
                    if os.path.isfile(pickle_file):
                        print("Loading data from {}...".format(pickle_file))
                        dataset = util.better_pickle.pickle_load(pickle_file)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

                        print("Loading data from {}...".format(gigaword_data_path))
                        data_loader = giga_word_loader.GigaWordLoader(data_path, tokenizer_name=tokenizer_name)

                        dataset = giga_word_dataset.GigaWordDataset(data_loader, tokenizer)
                        print("Pickling dataset to {} for faster loading next time...".format(pickle_file))
                        util.better_pickle.pickle_dump(dataset, pickle_file)
                        print("OK")
        return dataset

    @classmethod
    def create_model(cls, conf: config.Config) -> typing.Union[encore_model.EnCoreModel, entity_encoder.EntityEncoder]:
        """
        This creates the EnCoreModel that encapsulates all the modules specified in the experiment
        configuration.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
        Returns:
            model (:class:`encore_model.EnCoreModel`): The new EnCoreModel.
        """
        classifier = None
        encoder = None
        intermediate_layer = None
        model = None
        tokenizer = None

        encoder = cls._create_entity_encoder(conf)  # Creates an entity encoder.
        classifier_hidden_size = encoder.hidden_size()
        classifier = cls._create_classifier(conf, classifier_hidden_size)  # Creates a classifier.
        intermediate_layer = cls._create_intermediate_layer(conf)
        if conf.albert_enc_ent:
            tokenizer = AutoTokenizer.from_pretrained(experiment.ALBERT_XX_LARGE_VERSION)

        if conf.bert_enc_ent_cased:
            tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_CASED_VERSION)

        if conf.bert_enc_ent_uncased:
            tokenizer = AutoTokenizer.from_pretrained(experiment.BERT_BASE_UNCASED_VERSION)

        if conf.roberta_enc_ent:
            tokenizer = AutoTokenizer.from_pretrained(experiment.ROBERTA_LARGE_VERSION)

        if conf.pretrain:
            model = entity_encoder.EntityEncoder(classifier, encoder, intermediate_layer, tokenizer)
        else:
            model = encore_model.EnCoreModel(classifier, encoder, tokenizer)

        return model

    @classmethod
    def create_optimizer(
            cls,
            conf: config.Config,
            model: typing.Union[encore_model.EnCoreModel, entity_encoder.EntityEncoder]
    ) -> optim.Optimizer:
        """
        This creates an optimizer for training the model.
        Args:
            conf (::class:`config.Config`): The specified experiment configuration.
            model (::class:`encore_model.EnCoreModel`): The model to be trained.

        Returns:
            optimizer (::class:`optim.Optimizer`): The optimizer for training the model.
        """

        return optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=conf.learning_rate
        )

    @classmethod
    def generate_cl_labels(cls, clusters: typing.List[cluster.Cluster], pad_token: int):
        """

        Args:
            clusters:
            pad_token:
        Returns:

        """
        input_masks = []
        input_sequences = []
        labels = []
        for cluster_idx, current_cluster in enumerate(clusters):
            for sentence in current_cluster.sentences:
                if len(sentence.entity_mask) > 512:
                    """For sentences longer than 512, ignore labels for tokens that appear after 512."""
                    current_input_masks = []
                    current_input_sequences = []
                    current_labels = []
                    for idx, (mask, token) in enumerate(zip(sentence.entity_mask, sentence.token_ids)):
                        if idx <= 512:
                            current_input_masks.append(mask)
                            current_input_sequences.append(token)
                            if mask == 1:
                                current_labels.append(cluster_idx)
                    if sum(current_input_masks) == len(current_labels):
                        input_masks.append(current_input_masks)
                        input_sequences.append(current_input_sequences)
                        labels.append(current_labels)
                    else:
                        continue

                else:
                    if sum(sentence.entity_mask) > 0:
                        input_masks.append(sentence.entity_mask)
                        input_sequences.append(sentence.token_ids)
                        labels.append([cluster_idx] * len(sentence.label_placeholders))

        input_masks = cls.pad_sequences(input_masks, 0)
        input_sequences = cls.pad_sequences(input_sequences, pad_token)

        return input_sequences, input_masks, labels

    @classmethod
    def mask_sequences(
            cls,
            input_seq: torch.Tensor,
            mask_token_id: int,
            entity_mask: torch.Tensor = None,
            mask_percentage: float = 0.00
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        This masks a specified percentage of tokens in the input sequence.
        Args:
            input_seq:
            mask_token_id:
            entity_mask:
            mask_percentage:
        Returns:

        """
        # Sanitize args.
        insanity.sanitize_type("input_seq", input_seq, torch.Tensor)
        insanity.sanitize_type("mask_token_id", mask_token_id, int)
        insanity.sanitize_range("mask_token_id", mask_token_id, minimum=0)
        insanity.sanitize_type("mask_percentage", mask_percentage, float)
        insanity.sanitize_range("mask_percentage", mask_percentage, minimum=0.00)
        if entity_mask is not None:
            insanity.sanitize_type("entity_mask", entity_mask, torch.Tensor)
            # assert input_seq.shape == entity_mask.shape

        # Create labels, a clone of input_seq
        labels = input_seq.clone()

        # Sample tokens for masking.
        probability_matrix = torch.full(labels.shape, mask_percentage)

        masked_indices = None
        if entity_mask is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        else:
            # if probability_matrix.shape == entity_mask.shape:
            #     masked_indices = torch.bernoulli(probability_matrix).bool() & (~(entity_mask.bool()))
            # else:
            #     masked_indices = torch.bernoulli(probability_matrix).bool()
            masked_indices = entity_mask.bool()

        # labels[~masked_indices] = -100

        # input_seq[masked_indices] = mask_token_id

        return input_seq, labels

    @classmethod
    def pad_sequences(cls, input_seq: list, padding_token: typing.Union[str, int]) -> list:
        """

        Args:
            input_seq:
            padding_token:

        Returns:

        """
        max_len = max(len(s) for s in input_seq)
        return [s + [padding_token] * (max_len - len(s)) for s in input_seq]

    @classmethod
    def prepare_ontonotes_batch(
            cls,
            batch: typing.List[ontonotes_example.OntoNotesExample],
            labels: index_map.IndexMap,
            tokenizer: AutoTokenizer,
            full_span_mask: bool = False
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            batch:
            labels:
            tokenizer:
            full_span_mask:

        Returns:

        """
        # Generate tensors: entity mask, labels, input_ids, mask
        input_seqs = []
        entity_masks = []
        entity_labels = []

        for sample in batch:
            input_seq = sample.input_tokens
            entity_mask = [0] * len(input_seq)
            entity_mask[sample.entity_head_pos] = 1
            label_set = sample.entity_labels

            # For input seq, pre- and post-append the CLS and SEP tokens
            input_seq = [tokenizer.cls_token] + input_seq + [tokenizer.sep_token]
            entity_mask = [0] + entity_mask + [0]

            # Transform to ids.
            new_input_seq = []
            new_entity_mask = []

            for token, mask in zip(input_seq, entity_mask):
                new_tokens = tokenizer.tokenize(token)
                new_token_idx = tokenizer.convert_tokens_to_ids(new_tokens)
                if len(new_token_idx) > 1:
                    if mask == 1:
                        # TODO: Find the parent token and add a corresponding 1
                        new_mask = [0] * len(new_token_idx)
                        new_mask[0] = 1
                        new_entity_mask.extend(new_mask)
                    else:
                        new_entity_mask.extend([mask] * len(new_token_idx))

                    new_input_seq.extend(new_token_idx)
                else:
                    new_input_seq.extend(new_token_idx)
                    new_entity_mask.append(mask)
            label_idx = [labels.index(label) for label in label_set]
            current_labels = [0] * len(labels)
            for idx in label_idx:
                current_labels[idx] = 1

            input_seqs.append(new_input_seq)
            entity_masks.append(new_entity_mask)
            entity_labels.append(current_labels)

        padded_input_seqs = cls.pad_sequences(input_seqs, tokenizer.pad_token_id)
        padded_entity_masks = cls.pad_sequences(entity_masks, 0)

        input_seq = torch.LongTensor(padded_input_seqs)
        entity_mask = torch.LongTensor(padded_entity_masks)
        entity_labels = torch.LongTensor(entity_labels)

        input_seq, mlm_labels = cls.mask_sequences(input_seq, tokenizer.mask_token_id, entity_mask)

        return input_seq, mlm_labels, entity_mask, entity_labels
