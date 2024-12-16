"""
Back in university, I wrote a paper about why HuggingFace's SQuAD v2 metric is broken, and added my own metrics instead.
Here's what I wrote:

    HuggingFace’s evaluate module offers a SQuAD v1
    metric which, for a given set of predicted answer
    strings and reference answer strings, computes the
    unigram (i.e. word-by-word) exact match (EM) and
    F1 percentages.

    The same module also offers a SQuAD v2 metric
    which additionally measures an F1 score for the answerability
    classification task, judged based on the
    probability of a question being unanswerable and a
    threshold above which the question is classified as
    such. It purports to return the SQuAD v1 metrics
    as a subset of its output, but this is incorrect: span
    predictions are entirely ignored by this metric when
    the question is classified as unanswerable, even if
    it is answerable and the span is correct. This is
    unfair: good QA heads will receive a bad score
    due to a bad UA head. Hence, I re-implement the
    SQuAD v2 metric by keeping a clean separation
    between the two tasks.

I also documented the following quirks for the HuggingFace SQuAD v2 metric:
    - It assumes that there is a difference between the question being unanswerable and the actual answer being "".
      If you mess up this convention, the metric doesn't report its "NoAns" fields.

    - This gets even weirder: because HF forces you to give a prediction string (unlike the answer string), you would
      think that the only correct answer for an unanswerable question is "". Yet:
        - If your confidence is high, you can give it any non-empty string and it is accepted:
            - "aa" 0.9: good prediction
            - "" 0.9: good
            - "" 0.1: good
            - "aa" 0.1: bad prediction
        - If your confidence is low, an empty string is still accepted, and also, strings with stopwords count as empty strings:
            - "" 0.1: good prediction
            - "a" 0.1: good prediction
            - "the" 0.1: good prediction
            - "the a" 0.1: good prediction

    - If the actual answer is "" (so an answerable question with a zero-width answer), the confidence also affects everything.

-----

In total, I compute 18 metrics:
    - 14 for unanswerability, which is a binary task:
        - TP, TN, FP, FN
        - Accuracy
        - Class-macro Re, Pr, F1. Note that class-macro Re is also called "balanced accuracy", i.e. the mean accuracy
          across the sets of only positives and only negatives. https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
        - Re+, Pr+, F1+ and Re-, Pr-, F1-, where especially Re- and F1- are informative since SQuADv2 has a 67% class skew towards positives.
    - 2 for answerable examples:
        - Exact span match (EM): among the answerable questions, of those that you decide to predict a span for, how often is the full span correct?
                                 That is, EM == P(correct span | answerable AND predict answerable).
                                 This is a measure of how well your QA heads actually function. It does not take into
                                 account cases where you let the QA heads predict something despite knowing they will be
                                 wrong 100% guaranteed, nor where you refuse the QA heads to show what they can do despite
                                 knowing there is an answer 100% guaranteed.
        - Example-macro unigram F1
    - 2 for the combined task:
        - A correctness metric:
              P(correct) = P(correct, answerable) + P(correct, unanswerable)
                         = P(correct | answerable)P(answerable) + P(correct | unanswerable)P(unanswerable)
                         = P(correct span | ans, predict ans)*P(predict ans | ans)*P(ans) + P(predict unans | unans)P(unans)
                         = EM_ans*recall_ans*skew + recall_unans*(1-skew)
                         = EM_ans*(tp/p)*(p/N) + (tn/n)*(1-p/N)
                         = EM_ans*(tp/N) + 1*(tn/N)
        - Span EM adjusted for correct prediction of answerability:
              P(correct span | ans) = P(correct span | predict ans, ans)*P(predict ans | ans)
                                    + P(correct span | predict unans, ans)*P(predict unans | ans)
                                    = EM*recall_ans + 0*(1-recall_ans)
                                    = EM*recall_ans = EM*tp/p
              This is equivalent to EM for a system which defaults to always predicting a wrong span when the
              answerability classifier wrongfully assumes there is no answer.
"""
from typing import Dict, Set

import torch
import evaluate
from archit.instantiation.abstracts import Tensors, Tensor

from ._core import EvaluationEnvironment
from .cm import BinaryConfusionMatrix
from .qa import QA


class AQA(QA):
    """
    A better version of the SQuAD v2 (Answerability + Question Answering) metric.
    """

    def __init__(self, environment: EvaluationEnvironment, answerability_threshold: float=0.5):
        """
        Stores a confusion matrix with the following layout:
                                 predicted unanswerable      predicted answerable
        actually unanswerable    TN                          FP
        actually answerable      FN                          TP

        :param answerability_threshold: Predict answerable when you are at least this certain that it is answerable.
        """
        super().__init__(environment=environment)
        self.qa_true_positives = QA(environment)
        self.matrix = BinaryConfusionMatrix(threshold=answerability_threshold)

    def _addBatch(self, logits: Tensors, labels: Tensors,
                  input_ids: Tensor, context_mask: Tensor):
        qa_logits, ua_logits = logits
        qa_labels, ua_labels = labels

        # Individual QA and UA metrics
        qa_predictions, qa_answers = super()._addBatch(logits=qa_logits[torch.where(ua_labels)], labels=qa_labels[torch.where(ua_labels)],
                                                       input_ids=input_ids, context_mask=context_mask)  # Compute QA, but only for examples that have an answer. (Our prediction doesn't matter.)
        ua_predictions = self.matrix.add(logits=ua_logits, labels=ua_labels)  # If you didn't have a dedicated UA head, you could impute UA logits with the QA head when it says  start_index != end_index.

        # Joint metric: exact matches for those that were correctly predicted to be answerable
        qa_with_correct_ua = (torch.cumsum(ua_labels, dim=0)-1)[torch.where(ua_labels & ua_predictions)]
        self.qa_true_positives._addFromStrings(predictions=[qa_predictions[i] for i in qa_with_correct_ua],
                                               actual_answers=[qa_answers[i]  for i in qa_with_correct_ua])

    @classmethod
    def keys(cls) -> Set[str]:
        return {"correctness", "correctness-ans"} | \
               {"QA_Uni" + name for name in QA.keys()} | \
               {"UA_" + name for name in BinaryConfusionMatrix.keys()}

    def _finish(self) -> Dict[str, float]:
        # Compute the separate QA and UA metrics.
        qa_metrics = super()._finish()
        ua_metrics = self.matrix.computeFromMemory()

        # Combine them into one dictionary.
        metrics = {("QA_Uni" + key): value for key,value in qa_metrics.items()} | {("UA_" + key): value for key,value in ua_metrics.items()}

        # Compute the combined metrics.
        qa_tp_metrics = self.qa_true_positives._finish()

        # 1. Full correctness: for a negative, you have to predict it is negative. For a positive, you have to predict a correct span, which you only do when you also predict it is positive.
        #   P(correct) = P(correct, answerable) + P(correct, unanswerable)
        #              = P(correct | answerable)P(answerable) + P(correct | unanswerable)P(unanswerable)
        #              = P(correct span, predict ans | answerable)P(answerable) + P(predict unans | unanswerable)P(unanswerable)
        #              = P(correct span | ans, predict ans)*P(predict ans | ans)*P(ans) + P(predict unans | unans)P(unans)
        #              = EM_tp*recall_ans*skew + recall_unans*(1-skew)
        #              = (correct spans in TPs/tp)*(tp/p)*(p/N) + (tn/n)*(1-p/N)
        #              = (correct spans in TPs/N) + 1*(tn/N)
        actually_answerable = ua_metrics["TP"] + ua_metrics["FN"]
        total               = ua_metrics["TP"] + ua_metrics["FN"] + ua_metrics["TN"] + ua_metrics["FP"]
        EM_tp               = qa_tp_metrics["EM"]
        skew = actually_answerable/total
        re_p = ua_metrics["Re+"]
        re_n = ua_metrics["Re-"]

        metrics["correctness"] = (skew*re_p*EM_tp + (1-skew)*re_n)

        # 2. Full answerable correctness: in SQuAD v1, full correctness is equivalent to full correctness across answerable
        #                                 questions because there are no negatives. You predict a span for all examples and
        #                                 all examples have a span. In SQuAD v2, the same positive examples are used as in SQuAD v1,
        #                                 except now it is possible that your system wrongfully refuses to predict a span for
        #                                 it (equivalently, it predicts the wrong span). So, we adjust for that to get the new
        #                                 correctness rate when the system is presented with a positive.
        #   P(correct | ans) = P(correct span, predict ans | ans)
        #                    = P(correct span | ans, predict ans)*P(predict ans | ans)
        #                    = EM_tp*recall_ans
        # Note that EM == P(correct span | ans) measures the performance of the QA heads regardless of the UA heads.
        # The relation between it, EM_tp and the correctness computed here is
        #   EM = P(correct span | ans) = P(correct span, predict ans | ans) + P(correct span, predict unans | ans)
        #                              = P(correct span | predict ans, ans)*P(predict ans | ans) + P(correct span | predict unans, ans)*P(predict unans | ans)
        #                              = EM_tp*recall_ans + EM_fn*(1-recall_ans) >= EM_tp*recall_ans = P(correct | ans)
        # so we can say for sure that EM >= P(correct | ans), making the latter a harder metric.
        metrics["correctness-ans"] = EM_tp*re_p

        return metrics
