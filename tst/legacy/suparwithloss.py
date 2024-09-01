import torch
from supar.models.dep import BiaffineDependencyModel
from supar.utils.metric import AttachmentMetric


class SuparWithLoss(torch.nn.Module):
    """
    The supar BiaffineDependencyModel uses a different interface than a normal HuggingFace transformer, with, in particular,
    a .forward() method that doesn't return a loss, only the logits. It only offers loss with a separate method.

    This wrapper first generates the logits and then generates the loss.
    """

    def __init__(self, core: BiaffineDependencyModel):
        super().__init__()
        self.model = core
        self.base_model = core.encoder.model

    def forward(self,
                words: torch.LongTensor,
                labels_arcs=None, labels_rels=None,
                **kwargs) -> tuple:
        """
        Reimplementation of BiaffineDependencyParser.train_step(), which does inference to get logits, then creates a
        mask, and then computes loss.

        As seen here: https://github.com/yzhangcs/parser/blob/bebdd350e034c517cd5b71185e056503290164fa/supar/utils/field.py#L279
        the `words` tensor is not like input_ids since it has 3 dimensions: the batch dimension, the words dimension, and
        the subwords dimension. Separating those two allows subword pooling.
        """
        # Inference
        arc_scores, rel_scores = self.model(words)  # Builds its own attention mask by checking against the pad id.

        # Compute loss
        loss = None
        if labels_arcs is not None and labels_rels is not None:
            mask = torch.ones((words.size(0), words.size(1)), dtype=torch.bool, device=words.device)
            loss = self.model.loss(arc_scores, rel_scores, labels_arcs, labels_rels, mask, partial=True)

        if loss is None:
            return (arc_scores, rel_scores)
        else:
            return (loss, arc_scores, rel_scores)

    ####################################################################################################################

    def eval_step(self, words: torch.LongTensor,
                  labels_arcs: torch.LongTensor, labels_rels: torch.LongTensor) -> AttachmentMetric:
        """
        Shortened version of the eval_step. Kinda nice to have, except Trainer doesn't use it, so it's useless.
        """
        mask = labels_arcs.ge(0)
        arc_scores, rel_scores = self.model(words)
        loss                   = self.model.loss(arc_scores, rel_scores, labels_arcs, labels_rels, mask, partial=True)
        arc_preds, rel_preds   = self.model.decode(arc_scores, rel_scores, mask, self.args.tree, self.args.proj)  # TODO: There is a [1:] in this function that I don't like. Suspiciously hardcoded.
        return AttachmentMetric(loss, (arc_preds, rel_preds), (labels_arcs, labels_rels), mask)

    def evaluation(self, preprocessed_dataset):
        """
        Another method that is used nowhere but it would be so nice if anything did use it.
        """
        # TODO: It really could be this easy... The way you'd use this is to give the Trainer an empty DataLoader as
        #       evaluation set, yet somehow still trigger this on evaluation -- perhaps with a callback or something,
        #       but that callback would need access to the model.

        # FIXME: You need to collate into batches. Right now, `batch` is actually one example.
        return sum([self.eval_step(**batch) for batch in preprocessed_dataset], AttachmentMetric())
