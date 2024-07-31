from tktkt.preparation.mappers import TextMapper

from ..tasks._core import *


@dataclass
class PerturbDataset(Task):
    """
    Wrapper around a fine-tuning task that corrupts the text field of that task's dataset.
    Used for verifying how stable a model is against corrupted input.

    TODO: Should be able to specify the splits you want to apply this to. Sometimes you want to perturb only the
          training set, only the testing set, and so on.
    """

    def __init__(self, task: Task, perturbation: TextMapper, text_field_name: str):
        super().__init__(
            task_name=task.task_name,
            metric_config=task.metric_config,
            automodel_class=task.automodel_class,
            **task.automodel_args
        )
        self.method_implementations = task
        self.perturbation = perturbation
        self.text_field_name = text_field_name

    def loadDataset(self) -> DatasetDict:
        def perturbExample(example):
            original = example[self.text_field_name]
            if isinstance(original, str):
                example[self.text_field_name] = self.perturbation.convert(original)
            elif isinstance(original, list):
                example[self.text_field_name] = [self.perturbation.convert(word) for word in original]
            else:
                raise ValueError(f"Could not process text field '{self.text_field_name}': not a list or string.")
            return example

        return self.method_implementations.loadDataset().map(perturbExample, batched=False)

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        return self.method_implementations.prepareDataset(dataset)

    def getCollator(self) -> DataCollator:
        return self.method_implementations.getCollator()

    def getPredictionsAndReferences(self, eval: transformers.EvalPrediction) -> Tuple[Any,Any]:
        return self.method_implementations.getPredictionsAndReferences(eval)
