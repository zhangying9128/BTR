#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging

#zhangying
import json
from tqdm import tqdm
import torch.nn.functional as F

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    reset_logging()

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    #zhangying
    def _build_batchs(src_tokens, tgt_tokens, task):
        dataset = task.build_dataset_for_inference_BTR(src_tokens, [src.numel() for src in src_tokens], tgt_tokens, [tgt.numel() for tgt in tgt_tokens])
        batch = dataset.collater([ dataset[i] for i in range(dataset.__len__())])
        return batch

    with torch.no_grad():
        with open(os.path.join(cfg.task.test_cand_path, "output.tok.txt")) as f_pred:
            pred_data = f_pred.readlines()


        batch_scores = []
        for n in tqdm(range(len(pred_data))):
            pred = json.loads(pred_data[n])
            cand_srcs = []
            cand_tgts = []
            cand_idxs = []
            for idx, cand in enumerate(pred["gec_outputs_binary"][:cfg.common_eval.num_of_candidates]):
                if cand not in cand_tgts and len(cand) != 0:
                    cand_tgts.append(cand)
                    cand_idxs.append(idx)

            cand_srcs = [pred['source_strings_binary']+[task.target_dictionary.eos()]] * len(cand_tgts)
            cand_srcs = [torch.LongTensor(src) for src in cand_srcs]
            cand_tgts = [torch.LongTensor(tgt) for tgt in cand_tgts]

            if max([len(cand) for cand in cand_tgts]) <= 512:
                #build batch
                batch = _build_batchs(cand_srcs, cand_tgts, task)
                if torch.cuda.is_available():
                    batch = utils.move_to_cuda(batch)

                #predict
                hypos = []
                for i in range(len(batch)):
                    avg_probs = None
                    pred_list = []
                    for j in range(len(batch[i])):
                        sample = batch[i][j]
                        net_input = sample['net_input']
                        orig_target = sample['target']
                        target_positions = sample['target_positions']
                        decoder_attention_mask = sample['net_input']['prev_output_tokens'].ne(task.target_dictionary.pad()).int()
                        sample['net_input']['prev_output_tokens'][:, 0] = task.target_dictionary.pad()

                        decoder_out = model.decoder(sample['net_input']['src_tokens'], sample['net_input']['prev_output_tokens'], decoder_attention_mask)

                        bsz, _ , dim = decoder_out.size()
                        decoder_out = decoder_out.view(-1, dim)[sample['target_positions'].view(-1)].view(bsz, -1, dim)

                        curr_prob = F.log_softmax(decoder_out, dim=-1)
                        curr_prob = curr_prob.squeeze(1)
                        curr_prob = curr_prob.gather(dim=-1, index=orig_target)
                        for step in range(len(curr_prob)):
                            probs = curr_prob[step]
                            if avg_probs is None:
                                avg_probs = probs
                            else:
                                avg_probs = torch.cat((avg_probs, probs), dim=0)
                    score = avg_probs.sum() / avg_probs.size(0)

                    hypos.append([{
                        'score': score,
                        'positional_scores': avg_probs.view(-1),
                    }])


                scores = F.softmax(torch.FloatTensor([hypo[0]['positional_scores'].mean() for hypo in hypos]), dim=0)
                scores = scores.tolist() 
                _scores = [0] * cfg.common_eval.num_of_candidates
                for idx, score in zip(cand_idxs, scores):
                    _scores[idx] += score
            else:
                _scores = [1] + [0] * (cfg.common_eval.num_of_candidates-1)

            batch_scores.append(_scores)

        with open(cfg.common_eval.results_path, 'w') as f:
            for i in range(len(batch_scores)):
                f.write(json.dumps(batch_scores[i]) + '\n')
        logger.info("saving results to {}".format(cfg.common_eval.results_path))



def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()
