from ._general import CompareSentencesGLUETask


class STSB(CompareSentencesGLUETask):
    """
    Regressive tasks that tries to score sentence similarity between 0 and 5.
    """

    def __init__(self):
        super().__init__(task_name="stsb", num_labels=1, is_regressive=True)

    # Equivalent to argmax(-1) because there is only one class
    # def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
    #     predictions, labels = eval.predictions.squeeze(), eval.label_ids
    #     return predictions.tolist(), labels.tolist()
