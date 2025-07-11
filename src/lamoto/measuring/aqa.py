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
        - Re+, Pr+, F1+ and Re-, Pr-, F1-, where especially Re- and F1- are informative since SQuADv2 has a 67% class skew towards positives.
        - Class-macro Re, Pr, F1. Note that class-macro Re is also called "balanced accuracy", i.e. the mean accuracy
          across the sets of only positives and only negatives. https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
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
                         = P(correct span, predict ans | answerable)P(answerable) + P(predict unans | unanswerable)P(unanswerable)
                         = P(correct span | ans, predict ans)*P(predict ans | ans)*P(ans) + P(predict unans | unans)P(unans)
                         = EM_tp*recall_ans*skew + recall_unans*(1-skew)
        - Span EM adjusted for correct prediction of answerability:
              P(correct | ans) = P(correct span, predict ans | ans)
                               = P(correct span | ans, predict ans)*P(predict ans | ans)
                               = EM_tp*recall_ans
          This is equivalent to EM for a system which defaults to always predicting a wrong span when the
          answerability classifier wrongfully assumes there is no answer, which is never the case in SQuAD v1 and hence
          P(correct | ans) == EM == EM_tp in SQuAD v1.
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
        super().__init__(environment=environment)  # super() computes EM metric for all answerables.

        self.qa_true_positives = QA(environment)  # This computes EM metric only for the answerables that were not refused.
        self.matrix = BinaryConfusionMatrix(threshold=answerability_threshold)

    def _addBatch(self, logits: Tensors, labels: Tensors,
                  input_ids: Tensor, context_mask: Tensor):
        qa_logits, ua_logits = logits
        qa_labels, ua_labels = labels

        # Individual QA and UA metrics
        ans_mask = (ua_labels == 1)
        qa_predictions, qa_answers = super()._addBatch(logits=qa_logits[ans_mask], labels=qa_labels[ans_mask],
                                                       input_ids=input_ids, context_mask=context_mask)  # Compute QA, but only for examples that have an answer. (Our UA prediction doesn't matter.)
        ua_predictions = self.matrix.add(logits=ua_logits, labels=ua_labels)  # Footnote: If you didn't have a dedicated UA head, you could impute UA logits with the QA head when it says  start_index != end_index.

        # Joint metric: exact matches for those that were correctly predicted to be answerable
        indices_in_qa_with_correct_ua = (torch.cumsum(ua_labels, dim=0)-1)[ans_mask & ua_predictions]  # The cumsum-1 is a trick to get the enumerate indices of answerables (ua_labels == 1) but in a tensor that is still as large as the original batch. The values of this cumsum at unanswerable rows is bogus but will never be seen by the mask.
        self.qa_true_positives._addFromStrings(predictions=[qa_predictions[idx] for idx in indices_in_qa_with_correct_ua],
                                               actual_answers=[qa_answers[idx]  for idx in indices_in_qa_with_correct_ua])

    @classmethod
    def keys(cls) -> Set[str]:
        return {"correctness", "correctness-ans"} | \
               {"QA_Uni" + name for name in QA.keys()} | \
               {"UA_" + name    for name in BinaryConfusionMatrix.keys()}

    def _finish(self) -> Dict[str, float]:
        # Compute the separate QA and UA metrics.
        qa_metrics = super()._finish()
        ua_metrics = self.matrix.computeFromMemory()

        # Combine them into one dictionary.
        metrics = {("QA_Uni" + key): value for key,value in qa_metrics.items()} | \
                  {("UA_" + key): value for key,value in ua_metrics.items()}

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

        # 2. Full answerable correctness: for just the answerable questions, we can again measure if you correctly classify them and then find the span. This is easier than full correctness, because you aren't punished for viewing unanswerable questions as answerable.
        #   P(correct | ans) = P(correct span, predict ans | ans)
        #                    = P(correct span | ans, predict ans)*P(predict ans | ans)
        #                    = EM_tp*recall_ans
        metrics["correctness-ans"] = EM_tp*re_p

        # Two notes about this metric:
        #   1. About the difficulty as compared to EM:
        #       Because EM = P(correct span | ans) rather than P(correct | ans), it is always higher, because it does not require
        #       a correct answerability prediction, only a correct span prediction. This is easy to prove formally:
        #           EM = P(correct span | ans) = P(correct span, predict ans | ans) + P(correct span, predict unans | ans)
        #                                      = P(correct span | predict ans, ans)*P(predict ans | ans) + P(correct span | predict unans, ans)*P(predict unans | ans)
        #                                      = EM_tp*recall_ans + EM_fn*(1-recall_ans) >= EM_tp*recall_ans = P(correct | ans)
        #       which makes this reduced correctness still harder than EM.
        #   2. About comparing a SQuAD v2 system to a SQuAD v1 system:
        #       You may be tempted to compare the EM scores for SQuAD v1 and SQuAD v2 systems, because they have the same
        #       name. But actually, there are four metrics in SQuAD v2 that all collapse into one number for SQuAD v1:
        #       EM = P(correct span | ans), EM_tp = P(correct span | predict ans, ans), P(correct) and P(correct | ans).
        #       This is because in SQuAD v1, all questions are answerable, and all answerable questions have a span
        #       predicted for them (i.e. they are implicitly predicted to be answerable).
        #
        #       When running a SQuAD v2 system on the same dataset, it is possible that it wrongfully refuses to predict
        #       a span for an example, which in SQuAD v1 is equivalent to getting a non-exact match. That means you
        #       should actually compare SQuAD v2 to SQuAD v1 not using EM, but using P(correct | ans).
        #
        #       If you want to compare the QA heads of both systems, the EM metric is what you want.
        #       EM_tp can be higher or lower than EM and has no useful interpretation on its own.
        return metrics
