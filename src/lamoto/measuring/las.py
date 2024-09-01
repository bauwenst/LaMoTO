from typing import Any, Dict, Tuple

import torch
from supar.utils.metric import AttachmentMetric
from supar.models.dep.biaffine.model import CoNLL, MatrixTree, DependencyCRF

from ._core import LogitLabelMetric


class DependencyParsingMetrics(LogitLabelMetric):

    def __init__(self, _):
        super().__init__(None)
        self.content = AttachmentMetric()

    def add(self, other: AttachmentMetric):
        self.content += other

    def compute(self, predictions: Any, references: Any) -> Dict[str,Any]:
        """
        Ignore the input (because we know it's empty) and output your own internal metrics.
        """
        summary = {
            "uas": self.content.uas,
            "las": self.content.las,
            "ucm": self.content.ucm,
            "lcm": self.content.lcm
        }
        self.content = AttachmentMetric()
        return summary

    @staticmethod
    def logitsAndLabelsToMetric(logits: Tuple[torch.LongTensor,torch.LongTensor], labels: Tuple[torch.LongTensor,torch.LongTensor],
                                # criterion=torch.nn.CrossEntropyLoss(),
                                enforce_projective=False, enforce_tree=True) -> AttachmentMetric:
        """
        Static version of BiaffineDependencyParser's eval_step(), which is possible since it relies on loss() and decode()
        which are both technically static apart from needing self.criterion, which is a constant anyway.

        Actually, you can skip the loss. We're already tracking that since we have logits already.

        The logic in this function assumes that
            1. An extra dummy word was prepended to the word sequence, to function as a root head, masked out with -100 in the labels.
            2. Special tokens do not appear in the word sequence, only around it.
        The original implementation makes the same assumptions (they don't have a dummy word, but do assume exactly one [BOS] at the start).
        """
        arc_scores, rel_scores = logits
        arc_labels, rel_labels = labels
        mask = arc_labels.ge(0)

        # decode()
        arc_preds = arc_scores.argmax(-1)
        sequence_lengths = mask.sum(1)
        sequence_starts = (mask*1).argmax(-1)

        bad = [not CoNLL.istree(seq[start:start+length], enforce_projective)
               for start, length, seq in zip(sequence_starts.tolist(), sequence_lengths.tolist(), (arc_preds-sequence_starts[:,None]+1).tolist())]
        if enforce_tree and any(bad):  # Note: any examples flagged by [bad] will have their padding set to predicting 0. If you're wondering where those 0s come from, it's here, not any sort of mask.
            # FIXME: Note that if the below classes expect the root to be 0 instead of 1 (which is the case in our labels
            #        and what we train the scores to be), then any examples in [bad] will probably be fucked up by this process.
            #        (I have tested though, and it seems that with enforce_tree=True the evaluations are higher, so that
            #        indicates that 1-based trees are equally well-represented by MatrixTree.)
            if enforce_projective:
                arc_preds[bad] = DependencyCRF(arc_scores[bad], mask[bad].sum(-1)).argmax
            else:
                arc_preds[bad] = MatrixTree(arc_scores[bad], mask[bad].sum(-1)).argmax
        rel_preds = rel_scores.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        # loss()
        # s_arc, arcs = s_arc[mask], arcs[mask]
        # s_rel, rels = s_rel[mask], rels[mask]
        # s_rel = s_rel[torch.arange(len(arcs)), arcs]
        # arc_loss = criterion(s_arc, arcs)
        # rel_loss = criterion(s_rel, rels)
        # loss = arc_loss + rel_loss

        # This metric class only registers its constructor arguments when a loss is provided. Weird, but okay.
        return AttachmentMetric(0.0, (arc_preds, rel_preds), (arc_labels, rel_labels), mask)
