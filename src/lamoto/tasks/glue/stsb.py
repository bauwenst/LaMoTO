from transformers import EvalPrediction

from ._general import CompareSentencesGLUETask


class STSB(CompareSentencesGLUETask):
    """
    Regressive tasks that tries to score sentence similarity between 0 and 5.
    """

    def __init__(self):
        super().__init__(task_name="STS-B", num_labels=1, is_regressive=True)

    def getPredictionsAndReferences(self, eval: EvalPrediction):
        predictions, labels = eval.predictions.squeeze(), eval.label_ids
        return predictions.tolist(), labels.tolist()
