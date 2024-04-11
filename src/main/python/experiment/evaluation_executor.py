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

import typing

import expbase as xb
import expbase.util as util
import experiment
import experiment.component_factory as component_factory
import experiment.encore_model_checkpoint as encore_model_checkpoint
import experiment.entity_encoder_checkpoint as entity_encoder_checkpoint
import gc
import json
import numpy as np
import os
import shutil
import torch
import torch.utils.data as data
from sklearn.metrics import f1_score


class EvaluationExecutor(xb.EvaluationExecutor):
    """This implements the evaluation of considered model."""

    EVAL_PROTO = "eval.proto"
    """str: The name of the file in the results dir that is used to store the evaluation protocol."""

    MODEL_PREDICTIONS_MACRO_F1 = "macro_f1_predictions.proto"

    MODEL_PREDICTIONS_MICRO_F1 = "micro_f1_predictions.proto"

    PRETRAIN_EVAL_PROTO = "pretrain.proto"
    """str: The name of the file in the results dir that is used to store the evaluation protocol during pretraining."""

    @staticmethod
    def compute_f1(precision: float, recall: float) -> float:
        """
        This method computes F1 given precision and recall.
        Args:
            precision (float): The precision value.
            recall (float): The recall value.

        Returns:
            f1 (float): The computed f1 value.
        """
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def compute_macro_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> typing.Tuple[float, float]:
        """

        Args:
            predictions:
            targets:

        Returns:

        """
        macro_p = ((targets * predictions).sum(1) / predictions.sum(1)).mean()
        macro_r = ((targets * predictions).sum(1) / targets.sum(1)).mean()
        return macro_p, macro_r

    @staticmethod
    def compute_micro_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> typing.Tuple[float, float]:
        """

        Args:
            predictions:
            targets:

        Returns:

        """
        micro_p = (targets * predictions).sum() / predictions.sum()
        micro_r = (targets * predictions).sum() / targets.sum()
        return micro_p, micro_r

    def _run_evaluation(self) -> None:
        if self._conf.test and self._conf.checkpoints_dir:
            files = []
            for r, d, f in os.walk(self._conf.checkpoints_dir):
                for file in f:
                    if file.endswith(".ckpt"):
                        files.append(os.path.join(r, file))
            print("Number of checkpoints to evaluate: {}".format(len(files)))
            for ckpt_f in files:
                ckpt_path = ckpt_f

                # load the checkpoint to evaluate
                # ckpt_path = self._conf.checkpoint if self._conf.test else self._ckpt
                print("loading checkpoint from '{}'...".format(ckpt_path))
                ckpt = None
                if self._conf.pretrain:
                    ckpt = entity_encoder_checkpoint.EntityEncoderCheckpoint.load(ckpt_path)
                else:
                    ckpt = encore_model_checkpoint.EnCoreModelCheckpoint.load(ckpt_path)
                print("OK")
                print()
                # print header
                util.printing_tools.print_header("EVALUATION AFTER EPOCH {}".format(ckpt.epoch))
                eval_proto = None
                eval_proto_path = None

                if self._conf.pretrain:
                    eval_proto_path = os.path.join(self._conf.results_dir, self.PRETRAIN_EVAL_PROTO)
                else:
                    eval_proto_path = os.path.join(self._conf.results_dir, self.EVAL_PROTO)
                if os.path.isfile(eval_proto_path):
                    with open(eval_proto_path, "r") as f:
                        eval_proto = json.load(f)
                else:
                    if self._conf.ontonotes:
                        eval_proto = {
                            "all": [],
                            "best": {
                                "macro-f1": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "macro-p": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "macro-r": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "micro-f1": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "micro-p": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "micro-r": {
                                    "epoch": -1,
                                    "value": -np.inf
                                }
                            }
                        }
                    elif self._conf.pretrain:
                        eval_proto = {
                            "all": [],
                            "best": {
                                "Avg. loss": {
                                    "epoch": -1,
                                    "value": np.inf
                                }
                            }
                        }
                    else:
                        eval_proto = {
                            "all": [],
                            "best": {
                                "all-macro-f1": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "all-micro-f1": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "ent-macro-f1": {
                                    "epoch": -1,
                                    "value": -np.inf
                                },
                                "ent-micro-f1": {
                                    "epoch": -1,
                                    "value": -np.inf
                                }
                            }
                        }
                if self._conf.ontonotes:
                    dataset = component_factory.ComponentFactory.create_dataset(self._conf, test=True)
                    model = component_factory.ComponentFactory.create_model(self._conf)
                    model.load_state_dict(ckpt.model_state)
                    print("The total number of samples: {}".format(len(dataset)))
                    print("Starting evaluation...")
                    model.eval()

                    if self._conf.eval_gpu:
                        model.cuda()

                    predictions = []
                    targets = []
                    total_skipped = 0
                    for sample_idx, sample in enumerate(dataset):

                        input_seq, _, entity_mask, entity_labels = component_factory.ComponentFactory.prepare_ontonotes_batch(
                            sample,
                            dataset.entity_labels,
                            model.tokenizer,
                            self._conf.full_span_mask
                        )

                        # Move tensors to GPU if training is on GPU
                        if self._conf.eval_gpu:
                            input_seq = input_seq.cuda()
                            entity_mask = entity_mask.cuda()
                            entity_labels = entity_labels.cuda()

                        # Forward pass
                        print("***********")
                        print("Input seq: {}".format(input_seq.shape))
                        print("Entity mask: {}".format(entity_mask.shape))
                        print("Entity labels: {}".format(entity_labels.shape))
                        print("++++++++++++")
                        print()
                        try:
                            with torch.no_grad():
                                model_prediction = model.compute_top(input_seq, entity_mask)
                            model_prediction = model_prediction.to('cpu')
                            entity_labels = entity_labels.to('cpu')
                            predictions.append(model_prediction)
                            targets.append(entity_labels)
                            del input_seq
                            del model_prediction
                            del entity_labels
                            gc.collect()
                            torch.cuda.empty_cache()
                        except Exception as e:
                            print("Skipped sample {}/{}".format(sample_idx + 1, len(dataset)))
                            total_skipped += 1
                            print(e)
                            exit()
                        if (sample_idx + 1) % self._conf.print_int == 0:
                            print(
                                "Processed sample {}/{} ... Epoch {}".format(sample_idx + 1, len(dataset), ckpt.epoch))
                        # if (sample_idx + 1) == 2000:
                        #     print("Going to the next checkpoint")
                        #     break

                    # Compute OntoNotes metrics.
                    predictions = torch.cat(predictions, dim=0)
                    targets = torch.cat(targets, dim=0)

                    macro_precision, macro_recall = self.compute_macro_metrics(predictions, targets)
                    macro_f1 = self.compute_f1(macro_precision, macro_recall)

                    micro_precision, micro_recall = self.compute_micro_metrics(predictions, targets)
                    micro_f1 = self.compute_f1(micro_precision, micro_recall)

                    print("Macro-f1: {}".format(round(macro_f1.item(), 4)))
                    print("Micro-f1: {}".format(round(micro_f1.item(), 4)))
                    print("Total number of skipped samples: {}".format(total_skipped))
                    print()

                    # Update protocol
                    if eval_proto:
                        eval_proto["all"].append(
                            {
                                "epoch": ckpt.epoch,
                                "macro-f1": round(macro_f1.item(), 4),
                                "macro-p": round(macro_precision.item(), 4),
                                "macro-r": round(macro_recall.item(), 4),
                                "micro-f1": round(micro_f1.item(), 4),
                                "micro-p": round(micro_precision.item(), 4),
                                "micro-r": round(micro_recall.item(), 4)
                            }
                        )
                        if eval_proto["all"][-1]["macro-f1"] > eval_proto["best"]["macro-f1"]["value"]:
                            print("The previous best macro-f1 was improved.")

                            # Update protocol
                            eval_proto["best"]["macro-f1"]["value"] = eval_proto["all"][-1]["macro-f1"]
                            eval_proto["best"]["macro-f1"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".macro_f1")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["macro-p"] > eval_proto["best"]["macro-p"]["value"]:
                            print("The previous best macro-p was improved.")

                            # Update protocol
                            eval_proto["best"]["macro-p"]["value"] = eval_proto["all"][-1]["macro-p"]
                            eval_proto["best"]["macro-p"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".macro_p")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["macro-r"] > eval_proto["best"]["macro-r"]["value"]:
                            print("The previous best macro-r was improved.")

                            # Update protocol
                            eval_proto["best"]["macro-r"]["value"] = eval_proto["all"][-1]["macro-r"]
                            eval_proto["best"]["macro-r"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".macro_r")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["micro-f1"] > eval_proto["best"]["micro-f1"]["value"]:
                            print("The previous best micro-f1 was improved.")

                            # Update protocol
                            eval_proto["best"]["micro-f1"]["value"] = eval_proto["all"][-1]["micro-f1"]
                            eval_proto["best"]["micro-f1"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".micro_f1")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["micro-p"] > eval_proto["best"]["micro-p"]["value"]:
                            print("The previous best micro-p was improved.")

                            # Update protocol
                            eval_proto["best"]["micro-p"]["value"] = eval_proto["all"][-1]["micro-p"]
                            eval_proto["best"]["micro-p"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".micro_p")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["micro-r"] > eval_proto["best"]["micro-r"]["value"]:
                            print("The previous best micro-r was improved.")

                            # Update protocol
                            eval_proto["best"]["micro-r"]["value"] = eval_proto["all"][-1]["micro-r"]
                            eval_proto["best"]["micro-r"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".micro_r")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                elif self._conf.pretrain:
                    if eval_proto:
                        eval_proto["all"].append(
                            {
                                "epoch": ckpt.epoch,
                                "Avg. loss": ckpt.epoch_loss
                            }
                        )
                        # Update eval protocol
                        if eval_proto["all"][-1]["Avg. loss"] < eval_proto["best"]["Avg. loss"]["value"]:
                            print("The previous best Avg. loss was improved.")

                            # Update protocol
                            eval_proto["best"]["Avg. loss"]["value"] = eval_proto["all"][-1]["Avg. loss"]
                            eval_proto["best"]["Avg. loss"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".pretrain")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")
                else:
                    dataset = component_factory.ComponentFactory.create_dataset(self._conf, dev=True)
                    model = component_factory.ComponentFactory.create_model(self._conf)
                    model.load_state_dict(ckpt.model_state)
                    model.eval()
                    if self._conf.eval_gpu:
                        model.cuda()
                    predictions = []
                    predictions_ent_only = []
                    targets = []
                    targets_ent_only = []
                    total_num_samples = len(dataset)
                    for sample_idx, sample in enumerate(dataset):
                        input_seq, target_ent_labels, entity_mask = sample

                        # Convert lists to tensors.
                        entity_mask = torch.LongTensor(entity_mask).unsqueeze(0)
                        input_seq = torch.LongTensor(input_seq).unsqueeze(0)
                        target_labels = torch.LongTensor(target_ent_labels).unsqueeze(0)

                        # Call model
                        if self._conf.eval_gpu:
                            entity_mask = entity_mask.cuda()
                            input_seq = input_seq.cuda()
                            target_labels = targets.cuda()

                        output = model.compute_top(input_seq=input_seq)
                        prediction = torch.argmax(output, dim=2)
                        predictions += prediction.view(-1).tolist()
                        targets += target_labels.view(-1).tolist()

                        # Only entities
                        target_labels_ent = target_labels.view(-1).tolist()
                        target_labels_ent = [i for i, mask in zip(target_labels_ent, entity_mask.view(-1).tolist()) if
                                             mask == 1]
                        prediction_ent = prediction.view(-1).tolist()
                        prediction_ent = [i for i, mask in zip(prediction_ent, entity_mask.view(-1).tolist()) if
                                          mask == 1]
                        targets_ent_only += target_labels_ent
                        predictions_ent_only += prediction_ent
                        print("Processed sample: {}/{} ...".format(sample_idx + 1, total_num_samples))

                    all_macro_f1 = f1_score(targets, predictions, average='macro')
                    all_micro_f1 = f1_score(targets, predictions, average='micro')
                    ent_macro_f1 = f1_score(targets_ent_only, predictions_ent_only, average='macro')
                    ent_micro_f1 = f1_score(targets_ent_only, predictions_ent_only, average='micro')

                    if eval_proto:
                        eval_proto["all"].append(
                            {
                                "epoch": ckpt.epoch,
                                "all-macro-f1": round(all_macro_f1, 4),
                                "all-micro-f1": round(all_micro_f1, 4),
                                "ent-macro-f1": round(ent_macro_f1, 4),
                                "ent-micro-f1": round(ent_micro_f1, 4)
                            }
                        )
                        # Update eval protocol
                        if eval_proto["all"][-1]["all-macro-f1"] > eval_proto["best"]["all-macro-f1"]["value"]:
                            print("The previous best all-macro-f1 was improved.")

                            # Update protocol
                            eval_proto["best"]["all-macro-f1"]["value"] = eval_proto["all"][-1]["all-macro-f1"]
                            eval_proto["best"]["all-macro-f1"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".all_macro")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["all-micro-f1"] > eval_proto["best"]["all-micro-f1"]["value"]:
                            print("The previous best all-micro-f1 was improved.")

                            # Update protocol
                            eval_proto["best"]["all-micro-f1"]["value"] = eval_proto["all"][-1]["all-micro-f1"]
                            eval_proto["best"]["all-micro-f1"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".all_micro")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["ent-macro-f1"] > eval_proto["best"]["ent-macro-f1"]["value"]:
                            print("The previous best ent-macro-f1 was improved.")

                            # Update protocol
                            eval_proto["best"]["ent-macro-f1"]["value"] = eval_proto["all"][-1]["ent-macro-f1"]
                            eval_proto["best"]["ent-macro-f1"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".ent_macro")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        if eval_proto["all"][-1]["ent-micro-f1"] > eval_proto["best"]["ent-micro-f1"]["value"]:
                            print("The previous best ent-micro-f1 was improved.")

                            # Update protocol
                            eval_proto["best"]["ent-micro-f1"]["value"] = eval_proto["all"][-1]["ent-micro-f1"]
                            eval_proto["best"]["ent-micro-f1"]["epoch"] = ckpt.epoch

                            # store the current checkpoint
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".ent_micro")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                # update the protocol file
                with open(eval_proto_path, "w") as f:
                    json.dump(eval_proto, f, indent=4)

                # clean up the evaluated checkpoint, unless it is the final one, or we are in testing model
                if not self._conf.test and ckpt.epoch < self._conf.num_epochs - 1:
                    print("cleaning up checkpoint '{}'...".format(self._ckpt))
                    os.remove(self._ckpt)
                    print("OK")
                    print()

                # print footer
                util.printing_tools.print_end("EVALUATION AFTER EPOCH {}".format(ckpt.epoch))
        else:
            ckpt_path = self._conf.checkpoint if self._conf.test else self._ckpt
            print("loading checkpoint from '{}'...".format(ckpt_path))
            ckpt = None
            if self._conf.pretrain:
                ckpt = entity_encoder_checkpoint.EntityEncoderCheckpoint.load(ckpt_path)
            else:
                ckpt = encore_model_checkpoint.EnCoreModelCheckpoint.load(ckpt_path)
            print("OK")
            print()
            # print header
            util.printing_tools.print_header("EVALUATION AFTER EPOCH {}".format(ckpt.epoch))
            eval_proto = None
            eval_proto_path = None

            if self._conf.pretrain:
                eval_proto_path = os.path.join(self._conf.results_dir, self.PRETRAIN_EVAL_PROTO)
            else:
                eval_proto_path = os.path.join(self._conf.results_dir, self.EVAL_PROTO)
            if os.path.isfile(eval_proto_path):
                with open(eval_proto_path, "r") as f:
                    eval_proto = json.load(f)
            else:
                if self._conf.ontonotes or self._conf.figer:
                    eval_proto = {
                        "all": [],
                        "best": {
                            "macro-f1": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "macro-p": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "macro-r": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "micro-f1": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "micro-p": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "micro-r": {
                                "epoch": -1,
                                "value": -np.inf
                            }
                        }
                    }
                elif self._conf.pretrain:
                    eval_proto = {
                        "all": [],
                        "best": {
                            "Avg. loss": {
                                "epoch": -1,
                                "value": np.inf
                            }
                        }
                    }
                else:
                    eval_proto = {
                        "all": [],
                        "best": {
                            "all-macro-f1": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "all-micro-f1": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "ent-macro-f1": {
                                "epoch": -1,
                                "value": -np.inf
                            },
                            "ent-micro-f1": {
                                "epoch": -1,
                                "value": -np.inf
                            }
                        }
                    }
            if self._conf.ontonotes or self._conf.figer:
                macro_predictions_file = None
                macro_predictions_file_path = None
                micro_predictions_file = None
                micro_predictions_file_path = None
                if self._conf.print_predictions:
                    predictions_file = []
                    macro_predictions_file_path = os.path.join(self._conf.results_dir, self.MODEL_PREDICTIONS_MACRO_F1)
                    micro_predictions_file_path = os.path.join(self._conf.results_dir, self.MODEL_PREDICTIONS_MICRO_F1)

                dataset = component_factory.ComponentFactory.create_dataset(self._conf, test=True)
                model = component_factory.ComponentFactory.create_model(self._conf)
                model.load_state_dict(ckpt.model_state)
                print("The total number of samples: {}".format(len(dataset)))
                print("Starting evaluation...")
                model.eval()

                if self._conf.eval_gpu:
                    model.cuda()

                predictions = []
                targets = []
                total_skipped = 0
                for sample_idx, sample in enumerate(dataset):
                    input_seq, _, entity_mask, entity_labels = component_factory.ComponentFactory.prepare_ontonotes_batch(
                        sample,
                        dataset.entity_labels,
                        model.tokenizer,
                        self._conf.full_span_mask
                    )

                    # Move tensors to GPU if training is on GPU
                    if self._conf.eval_gpu:
                        input_seq = input_seq.cuda()
                        entity_mask = entity_mask.cuda()
                        entity_labels = entity_labels.cuda()

                    # Forward pass
                    try:
                        with torch.no_grad():
                            model_prediction, logits = model.compute_top(
                                input_seq,
                                entity_mask,
                                threshold=self._conf.prediction_threshold
                            )
                        model_prediction = model_prediction.to('cpu')
                        entity_labels = entity_labels.to('cpu')
                        predictions.append(model_prediction)
                        targets.append(entity_labels)
                        pred_list = model_prediction.view(-1).tolist()
                        target_list = entity_labels.view(-1).tolist()
                        prob_list = logits.view(-1).tolist()

                        if predictions_file is not None:
                            current_sample = {
                                "input sentence": sample[0].input_tokens,
                                "entity_head": sample[0].input_tokens[sample[0].entity_head_pos],
                                "gold_labels": [],
                                "prediction_confidence": []
                            }
                            for idx, (gold, pred, conf) in enumerate(zip(target_list, pred_list, prob_list)):
                                if gold == 1:
                                    current_sample["gold_labels"].append(dataset.entity_labels.value(idx))
                                    current_sample["prediction_confidence"].append(conf)

                            predictions_file.append(current_sample)
                        del input_seq
                        del model_prediction
                        del entity_labels
                        gc.collect()
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print("Skipped sample {}/{}".format(sample_idx + 1, len(dataset)))
                        total_skipped += 1
                    if (sample_idx + 1) % self._conf.print_int == 0:
                        print("Processed sample {}/{} ... Epoch {}".format(sample_idx + 1, len(dataset), ckpt.epoch))
                    # if (sample_idx + 1) == 2000:
                    #     print("Going to the next checkpoint")
                    #     break

                # Compute OntoNotes metrics.
                predictions = torch.cat(predictions, dim=0)
                targets = torch.cat(targets, dim=0)

                macro_precision, macro_recall = self.compute_macro_metrics(predictions, targets)
                macro_f1 = self.compute_f1(macro_precision, macro_recall)

                micro_precision, micro_recall = self.compute_micro_metrics(predictions, targets)
                micro_f1 = self.compute_f1(micro_precision, micro_recall)

                print("Macro-f1: {}".format(round(macro_f1.item(), 4)))
                print("Micro-f1: {}".format(round(micro_f1.item(), 4)))
                print("Total number of skipped samples: {}".format(total_skipped))
                print()

                # Update protocol
                if eval_proto:
                    eval_proto["all"].append(
                        {
                            "epoch": ckpt.epoch,
                            "macro-f1": round(macro_f1.item(), 4),
                            "macro-p": round(macro_precision.item(), 4),
                            "macro-r": round(macro_recall.item(), 4),
                            "micro-f1": round(micro_f1.item(), 4),
                            "micro-p": round(micro_precision.item(), 4),
                            "micro-r": round(micro_recall.item(), 4)
                        }
                    )
                    if eval_proto["all"][-1]["macro-f1"] > eval_proto["best"]["macro-f1"]["value"]:
                        print("The previous best macro-f1 was improved.")

                        # Update protocol
                        eval_proto["best"]["macro-f1"]["value"] = eval_proto["all"][-1]["macro-f1"]
                        eval_proto["best"]["macro-f1"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        if not self._conf.test:
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".macro_f1")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        # Update predictions file
                        if self._conf.print_predictions:
                            # Remove file if it exists
                            print("Remove existing predictions file...")
                            if os.path.isfile(macro_predictions_file_path):
                                os.remove(macro_predictions_file_path)
                            print("OK")
                            print("Create new predictions file ...")
                            with open(macro_predictions_file_path, "w") as f:
                                json.dump(predictions_file, f, indent=4)
                            print("OK")

                    if eval_proto["all"][-1]["macro-p"] > eval_proto["best"]["macro-p"]["value"]:
                        print("The previous best macro-p was improved.")

                        # Update protocol
                        eval_proto["best"]["macro-p"]["value"] = eval_proto["all"][-1]["macro-p"]
                        eval_proto["best"]["macro-p"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        if not self._conf.test:
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".macro_p")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                    if eval_proto["all"][-1]["macro-r"] > eval_proto["best"]["macro-r"]["value"]:
                        print("The previous best macro-r was improved.")

                        # Update protocol
                        eval_proto["best"]["macro-r"]["value"] = eval_proto["all"][-1]["macro-r"]
                        eval_proto["best"]["macro-r"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        if not self._conf.test:
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".macro_r")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                    if eval_proto["all"][-1]["micro-f1"] > eval_proto["best"]["micro-f1"]["value"]:
                        print("The previous best micro-f1 was improved.")

                        # Update protocol
                        eval_proto["best"]["micro-f1"]["value"] = eval_proto["all"][-1]["micro-f1"]
                        eval_proto["best"]["micro-f1"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        if not self._conf.test:
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".micro_f1")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                        # Update predictions file
                        if self._conf.print_predictions:
                            # Remove file if it exists
                            print("Remove existing predictions file...")
                            if os.path.isfile(micro_predictions_file_path):
                                os.remove(micro_predictions_file_path)
                            print("OK")
                            print("Create new predictions file ...")
                            with open(micro_predictions_file_path, "w") as f:
                                json.dump(predictions_file, f, indent=4)
                            print("OK")

                    if eval_proto["all"][-1]["micro-p"] > eval_proto["best"]["micro-p"]["value"]:
                        print("The previous best micro-p was improved.")

                        # Update protocol
                        eval_proto["best"]["micro-p"]["value"] = eval_proto["all"][-1]["micro-p"]
                        eval_proto["best"]["micro-p"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        if not self._conf.test:
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".micro_p")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

                    if eval_proto["all"][-1]["micro-r"] > eval_proto["best"]["micro-r"]["value"]:
                        print("The previous best micro-r was improved.")

                        # Update protocol
                        eval_proto["best"]["micro-r"]["value"] = eval_proto["all"][-1]["micro-r"]
                        eval_proto["best"]["micro-r"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        if not self._conf.test:
                            best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".micro_r")
                            print("storing current model to '{}'".format(best_path))
                            shutil.copyfile(self._ckpt, best_path)
                            print("OK")

            elif self._conf.pretrain:
                if eval_proto:
                    eval_proto["all"].append(
                        {
                            "epoch": ckpt.epoch,
                            "Avg. loss": ckpt.epoch_loss
                        }
                    )
                    # Update eval protocol
                    if eval_proto["all"][-1]["Avg. loss"] < eval_proto["best"]["Avg. loss"]["value"]:
                        print("The previous best Avg. loss was improved.")

                        # Update protocol
                        eval_proto["best"]["Avg. loss"]["value"] = eval_proto["all"][-1]["Avg. loss"]
                        eval_proto["best"]["Avg. loss"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".pretrain")
                        print("storing current model to '{}'".format(best_path))
                        shutil.copyfile(self._ckpt, best_path)
                        print("OK")
            else:
                dataset = component_factory.ComponentFactory.create_dataset(self._conf, dev=True)
                model = component_factory.ComponentFactory.create_model(self._conf)
                model.load_state_dict(ckpt.model_state)
                model.eval()
                if self._conf.eval_gpu:
                    model.cuda()
                predictions = []
                predictions_ent_only = []
                targets = []
                targets_ent_only = []
                total_num_samples = len(dataset)
                for sample_idx, sample in enumerate(dataset):
                    input_seq, target_ent_labels, entity_mask = sample

                    # Convert lists to tensors.
                    entity_mask = torch.LongTensor(entity_mask).unsqueeze(0)
                    input_seq = torch.LongTensor(input_seq).unsqueeze(0)
                    target_labels = torch.LongTensor(target_ent_labels).unsqueeze(0)

                    # Call model
                    if self._conf.eval_gpu:
                        entity_mask = entity_mask.cuda()
                        input_seq = input_seq.cuda()
                        target_labels = targets.cuda()

                    output = model.compute_top(input_seq=input_seq)
                    prediction = torch.argmax(output, dim=2)
                    predictions += prediction.view(-1).tolist()
                    targets += target_labels.view(-1).tolist()

                    # Only entities
                    target_labels_ent = target_labels.view(-1).tolist()
                    target_labels_ent = [i for i, mask in zip(target_labels_ent, entity_mask.view(-1).tolist()) if
                                         mask == 1]
                    prediction_ent = prediction.view(-1).tolist()
                    prediction_ent = [i for i, mask in zip(prediction_ent, entity_mask.view(-1).tolist()) if mask == 1]
                    targets_ent_only += target_labels_ent
                    predictions_ent_only += prediction_ent
                    print("Processed sample: {}/{} ...".format(sample_idx + 1, total_num_samples))

                all_macro_f1 = f1_score(targets, predictions, average='macro')
                all_micro_f1 = f1_score(targets, predictions, average='micro')
                ent_macro_f1 = f1_score(targets_ent_only, predictions_ent_only, average='macro')
                ent_micro_f1 = f1_score(targets_ent_only, predictions_ent_only, average='micro')

                if eval_proto:
                    eval_proto["all"].append(
                        {
                            "epoch": ckpt.epoch,
                            "all-macro-f1": round(all_macro_f1, 4),
                            "all-micro-f1": round(all_micro_f1, 4),
                            "ent-macro-f1": round(ent_macro_f1, 4),
                            "ent-micro-f1": round(ent_micro_f1, 4)
                        }
                    )
                    # Update eval protocol
                    if eval_proto["all"][-1]["all-macro-f1"] > eval_proto["best"]["all-macro-f1"]["value"]:
                        print("The previous best all-macro-f1 was improved.")

                        # Update protocol
                        eval_proto["best"]["all-macro-f1"]["value"] = eval_proto["all"][-1]["all-macro-f1"]
                        eval_proto["best"]["all-macro-f1"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".all_macro")
                        print("storing current model to '{}'".format(best_path))
                        shutil.copyfile(self._ckpt, best_path)
                        print("OK")

                    if eval_proto["all"][-1]["all-micro-f1"] > eval_proto["best"]["all-micro-f1"]["value"]:
                        print("The previous best all-micro-f1 was improved.")

                        # Update protocol
                        eval_proto["best"]["all-micro-f1"]["value"] = eval_proto["all"][-1]["all-micro-f1"]
                        eval_proto["best"]["all-micro-f1"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".all_micro")
                        print("storing current model to '{}'".format(best_path))
                        shutil.copyfile(self._ckpt, best_path)
                        print("OK")

                    if eval_proto["all"][-1]["ent-macro-f1"] > eval_proto["best"]["ent-macro-f1"]["value"]:
                        print("The previous best ent-macro-f1 was improved.")

                        # Update protocol
                        eval_proto["best"]["ent-macro-f1"]["value"] = eval_proto["all"][-1]["ent-macro-f1"]
                        eval_proto["best"]["ent-macro-f1"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".ent_macro")
                        print("storing current model to '{}'".format(best_path))
                        shutil.copyfile(self._ckpt, best_path)
                        print("OK")

                    if eval_proto["all"][-1]["ent-micro-f1"] > eval_proto["best"]["ent-micro-f1"]["value"]:
                        print("The previous best ent-micro-f1 was improved.")

                        # Update protocol
                        eval_proto["best"]["ent-micro-f1"]["value"] = eval_proto["all"][-1]["ent-micro-f1"]
                        eval_proto["best"]["ent-micro-f1"]["epoch"] = ckpt.epoch

                        # store the current checkpoint
                        best_path = os.path.join(self._conf.results_dir, experiment.BEST_CKPT_FILE + ".ent_micro")
                        print("storing current model to '{}'".format(best_path))
                        shutil.copyfile(self._ckpt, best_path)
                        print("OK")

            # update the protocol file
            with open(eval_proto_path, "w") as f:
                json.dump(eval_proto, f, indent=4)

            # clean up the evaluated checkpoint, unless it is the final one, or we are in testing model
            if not self._conf.test and ckpt.epoch < self._conf.num_epochs - 1:
                print("cleaning up checkpoint '{}'...".format(self._ckpt))
                os.remove(self._ckpt)
                print("OK")
                print()

            # print footer
            util.printing_tools.print_end("EVALUATION AFTER EPOCH {}".format(ckpt.epoch))
