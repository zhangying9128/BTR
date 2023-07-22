# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional, Iterator, List
from argparse import Namespace
from omegaconf import II
#zhangying
from collections import OrderedDict
import torch

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    LanguagePairDataset2, #zhangying
    RawLabelDataset, #zhangying
    iterators, #zhangying
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    bi_cand_file=None, discriminative_size=0, bidirectional_decoder=False, bidirectional_pretrain=False, #zhangying
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    cand_id_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )

        #zhangying
        if bi_cand_file is not None:
            cand_data = []
            with open(os.path.join(bi_cand_file, "output.tok.txt")) as f_pred:
                cand_data = f.readlines()

            e = MultiprocessingEdit(split)
            e.initializer(discriminative_size)
            for s, t, cands in zip(src_dataset._cache, tgt_dataset._cache, cand_data):
                src_cands, tgt_cands = e.load_cands((s, t, cands))
                for src_cand, tgt_cand in zip(src_cands, tgt_cands):
                    src_dataset.add_item(np.array(src_cand))
                    tgt_dataset.add_item(np.array(tgt_cand))
            cand_id = RawLabelDataset(tgt_dataset.cache_candid)
        else:
            cand_id = None

        cand_id_datasets.append(cand_id)

        src_datasets.append(src_dataset)
        if bidirectional_pretrain:
            src_datasets.append(tgt_dataset)
        if tgt_dataset is not None:
            if bidirectional_pretrain:
                tgt_datasets.append(src_dataset)
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )


        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        #zhangying
        cand_id_datasets = cand_id_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        #sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None
        #zhangying
        if cand_id_datasets[0] is not None:
            cand_id_datasets = ConcatDataset(cand_id_datasets, sample_ratios)
        else:
            cand_id_datasets = None

    if prepend_bos: #default False
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None: #default None
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id: #default False
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments: 
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        cand_id=cand_id_datasets,#zhangying
        bidirectional_decoder=bidirectional_decoder, #zhangying
        bidirectional_pretrain=bidirectional_pretrain, #zhangying
        split=split, #zhangying
    )

#zhangying
class MultiprocessingEdit(object):

    def __init__(self, split):
        self.split = split

    def initializer(self, discriminative_size):
        self.discriminative_size = discriminative_size


    def load_cands(self, input):
        src, tgt, cands = input
        gold = tgt.tolist()
        new_tgts = []
        pred = json.loads(cands)
        cands = pred["gec_outputs_binary"]
        _cands = OrderedDict()
        for cand in cands:
            cand.append(1)
            c = " ".join(map(str, cand))
            if c not in _cands and cand != gold:
                _cands[c] = cand
                if self.split == 'train':
                    if len(_cands) == min(len(src)-1, self.discriminative_size):
                        break
                else:
                    if len(_cands) == self.discriminative_size:
                        break

        new_tgts += list(_cands.values())[:self.discriminative_size]
        new_srcs = [src.tolist()] * len(new_tgts)
        return (new_srcs, new_tgts)


@dataclass
class BiDecoderTranslationConfig(FairseqDataclass):
    #------------zhangying----------
    do_evaluate: bool = field(
        default=False, metadata={"help": "do evaluation before training."}
    )
    bidirectional_pretrain: bool = field(
        default=False, metadata={"help": "pretrain the BTR"}
    )
    bidirectional_decoder: bool = field(
        default=False, metadata={"help": "use fully-visible self-attention in the decoder (i.e. bidirectional representation for the target sequence)."}
    )
    discriminative_size: int = field(
        default=0, metadata={"help": "number of candidates when using contrastive learning (i.e. negative sampling) during training."}
    )
    train_cand_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "the path contains candidates for training"
        },
    )
    valid_cand_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "the path contains candidates for validation"
        },
    )
    valid_m2: Optional[str] = field(
        default=None,
        metadata={
            "help": "the conll13 m2 file for validation"
        },
    )
    test_cand_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "the path contains candidates for testing"
        },
    )
    #------------zhangying----------


    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )


@register_task("bidecoder_translation", dataclass=BiDecoderTranslationConfig)
class BiDecoderTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: BiDecoderTranslationConfig

    def __init__(self, cfg: BiDecoderTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: BiDecoderTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # ------------------------------------------------------#
        # zhang ying
        bi_cand_file = None
        discriminative_size = 0
        if self.cfg.discriminative_size > 0:
            if split == "train" and self.cfg.train_cand_path is not None: 
                bi_cand_file = self.cfg.train_cand_path
                discriminative_size = self.cfg.discriminative_size
            
            if split == "valid" and self.cfg.valid_cand_path is not None:
                bi_cand_file = self.cfg.valid_cand_path
                discriminative_size = 5

        dataset_impl = 'mmapcached' if bi_cand_file is not None else self.cfg.dataset_impl
        
        # ------------------------------------------------------#


        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            bi_cand_file=bi_cand_file, #zhangying
            discriminative_size=discriminative_size, #zhanging
            bidirectional_decoder=self.cfg.bidirectional_decoder, #zhangying
            bidirectional_pretrain=self.cfg.bidirectional_pretrain, #zhangying
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    #zhangying
    def build_dataset_for_inference_BTR(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
        return LanguagePairDataset2(
            src_tokens, src_lengths, self.source_dictionary,
            tgt_tokens, tgt_lengths, self.target_dictionary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            append_eos_to_target=True,
        )


    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    #zhangying
    def get_test_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        dataset.index_sort = True

        # get indices ordered by example size
        indices = dataset.ordered_indices()


        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
            )


        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        return epoch_iter

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
        split='train', #zhangying
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        #zhangying
        logger.info("{} indices are available".format(len(indices)))

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter