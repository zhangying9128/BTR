# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class BidecoderCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("bidecoder_cross_entropy", dataclass=BidecoderCrossEntropyCriterionConfig)
class BidecoderCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self._has_logged_once = False
        self.sentence_avg = sentence_avg
        self.bidirectional_decoder = task.cfg.bidirectional_decoder

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #net_output = model(**sample["net_input"])

        #zhangying
        decoder_attention_mask = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).int()
        sample['net_input']['prev_output_tokens'][:, 0] = self.padding_idx

        logits = model.decoder(sample['net_input']['src_tokens'], sample['net_input']['prev_output_tokens'], decoder_attention_mask)
        if sample['target'].size() != sample['net_input']['prev_output_tokens'].size():
            bsz, _, dim = logits.size()
            logits = logits.view(-1, dim)[sample['target_positions'].view(-1)].view(bsz, -1, dim)

        sample['ntokens'] = sample['target'].size(0) * sample['target'].size(1)

        if 'cand_id' in sample:
            loss = self.compute_discriminative_loss(logits, sample['target'], sample['cand_id'])
        else:
            loss = self.compute_loss(logits, sample['target'], reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def compute_loss(self, logits, target, reduce=False):
        try:
            import xentropy_cuda
            from apex.contrib import xentropy

            if not self._has_logged_once:
                logger.info("using fused cross entropy")
                self._has_logged_once = True

            half_to_float = logits.dtype == torch.half
            losses = xentropy.SoftmaxCrossEntropyLoss.apply(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                0.0,
                self.padding_idx,
                half_to_float,
            )
            if reduce:
                return losses.sum()
            else:
                return losses

        except ImportError:
            lprobs = F.log_softmax(logits, dim=-1)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            loss = F.nll_loss(
                lprobs,
                target.view(-1),
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            return loss

    #zhangying
    def compute_discriminative_loss(self, logits, targets, cand_id):
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        targets = targets.unsqueeze(2)
        target_mask = (targets != self.padding_idx)
        probs = probs.gather(dim=-1, index=targets)
        cand_index = cand_id.nonzero(as_tuple=False)
        probs[cand_index] = 1 - probs[cand_index]
        probs = torch.clamp(probs, 1e-12, 1) #1e-12
        lprobs = - torch.log(probs)
        discriminative_loss = lprobs.masked_select(target_mask)
        discriminative_loss = discriminative_loss.sum()
        return discriminative_loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
