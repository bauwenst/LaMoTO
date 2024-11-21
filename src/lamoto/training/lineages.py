"""
You can use LaMoTO for one-off model training. For research, however, we often compare models. This is done by first
pretraining given configurations, and then taking each checkpoint and fine-tuning them on multiple tasks.

Although you could see all of these experiments as entirely separate, it makes more sense to speak in terms of an
"ancestry" or a "lineage" of models. Within one lineage, the model backbone and the tokeniser always stay the same, but
the weights change, the hyperparameters change, and the head changes.
When you want to set up training experiments from the command line, rather than having to declare the tokeniser and
checkpoint on the command line, it's nicer to be able to just request to "train node X in lineage A" and have the backend
load checkpoints and tokenisers for you.
"""
from typing import Type, List, Iterable, Union, Optional, Self
from transformers import PretrainedConfig
from abc import ABC, abstractmethod
from pathlib import Path
Checkpoint = Union[str,Path]

from tktkt.builders.base import TokeniserBuilder
from archit.instantiation.abstracts import BaseModel

from .auxiliary.hyperparameters import TaskHyperparameters
from .core import TaskTrainer, Task
from .tuning import TaskTuner, MetaHyperparameters


class LineageNode(ABC):
    """
    One operation that advances a lineage.
    """

    def __init__(self, handle: str, hp: TaskHyperparameters, out: Optional[Checkpoint]=None):
        """
        Some of the given hyperparameters will be ignored, e.g. the base model type and the tokeniser, and for
        finetuning, every hyperparameter in the grid.

        :param out: Checkpoint identifier indicating the RESULT of running this lineage node.
                    You leave this parameter None when you run your experiment, and then fill it in when you want to
                    do follow-up experiments.
        """
        self._handle = handle
        self._out    = out
        self._hp     = hp

        self._children: List[LineageNode] = []
        self._parent: Optional[LineageNode] = None

    def followedBy(self, child: "LineageNode"):
        if child._parent is not None:
            raise RuntimeError(f"Node '{child._handle}' already has a parent '{child._parent._handle}', so cannot make '{self._handle}' the parent.")

        self._children.append(child)
        child._parent = self

    @abstractmethod
    def duplicate(self) -> Self:
        """Get an object of the same subclass that shares all its fields with the current object, but with no child/parent relationships."""
        pass

    @abstractmethod
    def _run(self):
        """Do what this node should do to generate its output checkpoint."""
        pass

    def _buildHyperparameters(self) -> TaskHyperparameters:
        """
        Retrieve the node's hyperparameters while imputing the tokeniser and base model from the lineage this node belongs
        to, and imputing the checkpoint from the parent node.
        """
        # Find the lineage by tracing back up to the starting node.
        node = self
        while not isinstance(node, _StartingNode):
            node = node._parent
        assert isinstance(node, _StartingNode)
        lineage = node._lineage

        # Copy HPs because e.g. storing the tokeniser permanently in this node's HPs is a bad idea.
        hp = self._hp.copy()
        hp.archit_basemodel_class = lineage.base_model
        hp.TOKENISER              = lineage.tokeniser.buildTokeniser()
        hp.MODEL_CONFIG_OR_CHECKPOINT = self._parent._out
        return hp

    def __iter__(self):
        yield self
        for node in self._children:
            yield from node.__iter__()


class Lineage:
    """
    Ancestral tree of checkpoints. A basic tree may look something like:
    ```
                                                                  /---[tuning node]--> fine-tuned model
        [start]--> config --[training node]--> pretrained model --|---[tuning node]--> fine-tuned model
                                                                  \---[tuning node]--> fine-tuned model
    ```
    """

    def __init__(self, name: str, tokeniser: TokeniserBuilder, base_model: Type[BaseModel],
                 starting_config_or_checkpoint: Union[Checkpoint, PretrainedConfig], tree: LineageNode):
        self.name = name
        self.tokeniser = tokeniser
        self.base_model = base_model

        self._nodes = _StartingNode(self, starting_config_or_checkpoint)
        self._nodes.followedBy(tree)
        assert len(list(self.whichHandles())) == len(set(self.whichHandles())), "Lineage contains at least two nodes with the same handle."

    def __iter__(self) -> Iterable["LineageNode"]:
        yield from self._nodes.__iter__()

    def run(self, node_handle: str):
        for node in self:
            if node._handle == node_handle:
                node._run()
                break
        else:
            raise ValueError(f"Handle not in lineage: {node_handle}")

    def whichHandles(self) -> List[str]:
        return [node._handle for node in self]


class LineageRegistry:
    """
    Collection of all the lineages in your project.
    """

    def __init__(self):
        self._registry = dict()

    def add(self, lineage: Lineage):
        self._registry[lineage.name] = lineage

    def get(self, name: str) -> Lineage:
        return self._registry.get(name)

    def __iter__(self) -> Iterable["Lineage"]:
        for lineage in self._registry.values():
            yield lineage


class _StartingNode(LineageNode):

    def __init__(self, lineage: Lineage, model_config_or_checkpoint: Union[Checkpoint, PretrainedConfig]):
        super().__init__("", hp=None, out=model_config_or_checkpoint)
        self._lineage = lineage

    def _run(self):
        raise RuntimeError()


class TrainingNode(LineageNode):

    def __init__(self, handle: str, hp: TaskHyperparameters, trainer: TaskTrainer, task: Task, out: Optional[Checkpoint]=None):
        super().__init__(handle=handle, hp=hp, out=out)
        self._trainer = trainer
        self._task = task

    def _run(self):
        result = self._trainer.train(task=self._task, hyperparameters=self._buildHyperparameters())
        self._task.resetTemporaryFields()
        self._task.resetCaches()
        return result

    def duplicate(self) -> Self:
        return TrainingNode(self._handle, self._hp, self._trainer, self._task)


class TuningNode(LineageNode):

    def __init__(self, handle: str, hp: TaskHyperparameters, meta: MetaHyperparameters, tuner: TaskTuner, task: Task, out: Optional[Checkpoint]=None):
        super().__init__(handle=handle, hp=hp, out=out)
        self._tuner = tuner
        self._task = task
        self._meta = meta

    def _run(self):
        result = self._tuner.tune(task=self._task, hp=self._buildHyperparameters(), meta=self._meta)
        self._task.resetTemporaryFields()
        self._task.resetCaches()
        return result

    def duplicate(self) -> Self:
        return TuningNode(self._handle, self._hp, self._meta, self._tuner, self._task)