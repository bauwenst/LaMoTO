from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
from supar.utils.metric import AttachmentMetric
from transformers import AutoTokenizer, RobertaTokenizer, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.auto.auto_factory import _BaseAutoModelClass

# Note: You must install supar from GitHub. The pip version is more than 2 years out of date! https://github.com/yzhangcs/parser
from supar.models.dep.biaffine.model import BiaffineDependencyModel, CoNLL, MatrixTree, DependencyCRF
from supar.models.dep.biaffine.parser import BiaffineDependencyParser
from supar.utils.fn import pad


def relu(x):
    return max(0,x)


@dataclass
class DataCollatorForDependencyParsing(DataCollatorMixin):
    """
    Takes care of two preprocessing responsibilities:
        - Since dependency parsing has two label sequences instead of one, it pads both of them, not just one.
        - Handles 3D padding instead of 2D padding. Since the supar dependency parser uses token-to-word pooling, it
          expects not a 2D (batch x subwords) input_ids tensor but a 3D one (batch x words x subwords).
          This also includes truncating token sequences for one word that are too long. Truncation is normally handled
          by the tokeniser, but it makes more sense to do truncation and padding in the same place.

    What is expected of the incoming examples:
        - They should individually already be ready to be fed to a transformer. That means a tokeniser must have already
          converted strings to integer IDs.
        - Special tokens have to be added already and the labels must reflect this by having a -100 for those words.

    Why have this collator if supar already works, so it has some kind of built-in collator?
        - supar doesn't allow choosing the special token template, it does it for the user, no matter the tokeniser.
        - supar pads its labels somewhere deep inside obscure classes. The labels are padded when a list of sentences
          is converted to a batch, since Batch.compose calls Field.compose which is just a redirect to the pad() function.
          Batch.compose is called inside the "PrefetchGenerator" that underlies supar's DataLoader
          (https://github.com/yzhangcs/parser/blob/bebdd350e034c517cd5b71185e056503290164fa/supar/utils/data.py#L343).
          So in short: yeah there is some kind of collation happening, but holy fuck is it hidden deep.
    """

    tokenizer: PreTrainedTokenizerBase
    max_subwords_per_word: int=None
    label_pad_value: int = -100
    return_tensors: str = "pt"

    def torch_call(self, examples: List[dict]):
        # First handle the tokens.
        #   - Truncate words with many tokens, and pad those with few tokens.
        for example in examples:
            subword_dim = min(self.max_subwords_per_word or 1e99, max(map(len, example["words"])))  # Keep the per-example tensors as small as possible.
            example["words"] = [subwords[:subword_dim] + relu(subword_dim - len(subwords))*[self.tokenizer.pad_token_id]
                                for subwords in example["words"]]

        #   - Take tensors of size words_1 x max(tokens_1), ..., words_n x max(tokens_n) and turn them into a single tensor
        #     n x max(words_i) x max(max(tokens_i)_i)
        ids_tensor = pad([torch.tensor(example["words"], dtype=torch.long) for example in examples], self.tokenizer.pad_token_id)

        # For the labels, we assume there is going to be at least one sequence that has as many labels as ids_tensor.size(1).
        # (Because we assume that for all examples, len(example["words"]) == len(example["labels_arcs"]) == len(example["labels_rels"])).
        labels_arcs = pad([torch.tensor(example["labels_arcs"], dtype=torch.long) for example in examples], self.label_pad_value)
        labels_rels = pad([torch.tensor(example["labels_rels"], dtype=torch.long) for example in examples], self.label_pad_value)
        assert ids_tensor.size(1) == labels_arcs.size(1) == labels_rels.size(1)

        return {
            "words": ids_tensor,
            "labels_arcs": labels_arcs,
            "labels_rels": labels_rels
        }


class AutoModelForDependencyParsing(_BaseAutoModelClass):
    """
    Wrapper around supar to instantiate one specific type of dependency parser, namely a biaffine head (not a CRF)
    with a transformer backbone (not an LSTM).
    """

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs):
        tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = BiaffineDependencyModel(
            # Model arguments
            encoder="bert",  # "bert" here means "any MLM checkpoint". Could've been "not-lstm" too.
            bert=checkpoint,
            n_bert_layers=1,  # This is the amount of final hidden layers to mix together. 4 is the standard if you do any mixing.
            bert_pooling="mean",  # How to pool tokens of one word.
            pad_index=tokenizer.pad_token_id,
            unk_index=tokenizer.unk_token_id,
            # mix_dropout=,
            # n_encoder_hidden=,  # Has a default of 800, but for non-LSTMs, this argument is manually overridden by n_out, which is the size of the embeddings after passing through a linear transform on top of the encoder. Yet, supar's BiaffineDependencyModel constructor doesn't actually allow controlling n_out, and it defaults to the embedding size, which also causes the projection to be left out (rather than having a square projection).

            # Head arguments (these are the supar defaults)
            n_arc_mlp=500,
            n_rel_mlp=100,
            mlp_dropout=0.33,
            scale=0,
            n_rels=kwargs.pop("num_labels"),

            # Argument that doesn't have a default yet is only used for LSTMs.
            n_words=-1,
            **kwargs
        )
        return SuparWithLoss(model)


@dataclass
class DependencyParsingOutput:
    loss: Optional[torch.FloatTensor] = None
    arc_scores: torch.FloatTensor = None
    relation_scores: torch.FloatTensor = None


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
                return_dict: bool=False,
                **kwargs) -> DependencyParsingOutput:
        """
        As seen here: https://github.com/yzhangcs/parser/blob/bebdd350e034c517cd5b71185e056503290164fa/supar/utils/field.py#L279
        the `words` tensor is not like input_ids since it has 3 dimensions: the batch dimension, the words dimension, and
        the subwords dimension. Separating those two allows subword pooling.
        """
        # Inference
        arc_scores, rel_scores = self.model(words)  # Builds its own attention mask by checking against the pad id.

        # Compute loss
        loss = None
        if labels_arcs is not None and labels_rels is not None:
            mask = torch.ones((words.size(0), words.size(1)), dtype=torch.bool, device=words.device)  # You can simply initialise the mask to a full-1 matrix, since the data preprocessor already put a -100 on all padding and special tokens.
            # mask = batch.mask
            # mask[:, 0] = 0  # ignore the first token of each sentence  TODO: Why? --> My guess: they're assuming BoS tokens EoS format, and using the EoS as dummy for the root head.
            loss = self.model.loss(arc_scores, rel_scores, labels_arcs, labels_rels, mask, partial=True)  # The "partial" lets supar generate the mask using -100 labels.

        if not return_dict:  # It makes more sense to use dataclasses, but Trainer assumes the tuple output convention instead.
            if loss is None:
                return (arc_scores, rel_scores)
            else:
                return (loss, arc_scores, rel_scores)
        else:
            return DependencyParsingOutput(
                loss=loss,
                arc_scores=arc_scores,
                relation_scores=rel_scores
            )

    @staticmethod
    def logitsAndLabelsToMetric(logits: Tuple[torch.LongTensor,torch.LongTensor], labels: Tuple[torch.LongTensor,torch.LongTensor],
                                # criterion=torch.nn.CrossEntropyLoss(),
                                enforce_projective=False, enforce_tree=True) -> AttachmentMetric:
        """
        Static version of BiaffineDependencyModel's eval_step(), which is possible since it relies on loss() and decode()
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
        # TODO: It really could be this easy... The way you'd use this is to give the Trainer an empty DataLoader as
        #       evaluation set, yet somehow still trigger this on evaluation -- perhaps with a callback or something,
        #       but that callback would need access to the model.

        # FIXME: You need to collate into batches. Right now, `batch` is actually one example.
        return sum([self.eval_step(**batch) for batch in preprocessed_dataset], AttachmentMetric())
