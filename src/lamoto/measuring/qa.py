from typing import Dict, Set, Tuple, List
from torch import Tensor

import torch
import evaluate

from ._core import AutonomousMetric, EvaluationEnvironment


class QA(AutonomousMetric):
    """
    Wrapper around HuggingFace's SQuAD metric because it requires text whilst we have logits, labels and input_ids.
    The metric is autonomous because you need access to the actual evaluation text itself, not just the logits across it.

    TODO: May also want to add the "GRIM" metric: https://arxiv.org/abs/2206.14348
    """

    def __init__(self, environment: EvaluationEnvironment):
        super().__init__(environment=environment)
        self.squad_counter = 0
        self.squad_v1 = evaluate.load("squad")

    def computeFromEnvironment(self) -> Dict[str, float]:
        for batch in self.environment.getDatasetWithCollator():  # Unlike in PPL calculation, you can have batched inference here.
            context_mask = batch.pop("context_mask")

            with torch.no_grad():
                outputs = self.environment.model(**batch)
                self._addBatch(outputs, batch["labels"],
                               batch["input_ids"], context_mask)

        return self._finish()

    def _addBatch(self, logits: Tensor, labels: Tensor,
                  input_ids: Tensor, context_mask: Tensor) -> Tuple[List[str],List[str]]:
        predicted_answers = []
        actual_answers    = []

        # === PART 1: Batch operations ===
        # Disentangle start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)  # B x L x 2  ->  B x L and B x L.
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits   = end_logits  .squeeze(-1).contiguous()

        # Mask out the question
        start_logits = torch.masked_fill(start_logits, ~context_mask, -10_000)  # Everything not in context (namely the question, specials, and padding) gets probability ~e^{-10000} of being start/end.
        end_logits   = torch.masked_fill(end_logits,   ~context_mask, -10_000)

        # Softmax across the tokens
        start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
        end_probabilities   = torch.nn.functional.softmax(end_logits, dim=-1)

        # === PART 2: Per-example operations ===
        n_examples = start_logits.shape[0]  # Can be < BATCH_SIZE at the end.
        n_tokens   = start_logits.shape[1]
        for example_idx in range(n_examples):
            # Triangular outer product to get joint span probabilities (https://huggingface.co/course/chapter6/3b?fw=pt)
            start_probas = start_probabilities[example_idx]
            end_probas   = end_probabilities[example_idx]
            scores = torch.outer(start_probas, end_probas)  # Same as [:, None] * [None, :] but way clearer.
            scores = torch.triu(scores)

            index_of_max = scores.argmax().item()  # index given as if the scores matrix is serialised.
            predicted_start_index = index_of_max // n_tokens
            predicted_end_index   = index_of_max % n_tokens

            actual_start_index, actual_end_index = labels[example_idx]

            all_tokens = input_ids[example_idx]
            predicted_answer = self.environment.tokeniser.decode(all_tokens[predicted_start_index:predicted_end_index+1], skip_special_tokens=True).strip()
            actual_answer    = self.environment.tokeniser.decode(all_tokens[actual_start_index:actual_end_index+1],       skip_special_tokens=True).strip()

            # At this point, we have everything and it's just a matter of marshalling the results to feed them to HF.
            predicted_answers.append(predicted_answer)
            actual_answers.append(actual_answer)

        self._addFromStrings(predicted_answers, actual_answers)
        return predicted_answers, actual_answers

    def _addFromStrings(self, predictions: List[str], actual_answers: List[str]):
        for predicted_answer, actual_answer in zip(predictions, actual_answers):
            prediction = {"id": str(self.squad_counter), "prediction_text": predicted_answer}
            reference  = {"id": str(self.squad_counter), "answers": [{"text": actual_answer, "answer_start": 0}]}  # The metric doesn't use this 0, so it doesn't matter.
            self.squad_v1.add(prediction=prediction, reference=reference)
            self.squad_counter += 1  # If these IDs are not unique, the squad_metric completely messes up.

    @classmethod
    def keys(cls) -> Set[str]:
        return {"EM", "F1"}

    def _finish(self) -> Dict[str, float]:
        raw_metrics = self.squad_v1.compute()
        self.squad_counter = 0

        metrics = dict()
        metrics["EM"] = raw_metrics["exact_match"] / 100
        metrics["F1"] = raw_metrics["f1"] / 100
        return metrics
