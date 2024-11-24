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

from archit.instantiation.abstracts import BaseModel
from tktkt.builders.base import TokeniserBuilder
from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
SerialisedTokeniser = Union[str,TokeniserBuilder[TokeniserWithFiniteTypeDomain]]

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
        self.handle = handle
        self.out    = out
        self._hp    = hp

        self._children: List[LineageNode] = []
        self._parent: Optional[LineageNode] = None

    def followUp(self, child: "LineageNode") -> "LineageNode":
        """Add a node to the list of nodes that use the output of this node as their starting checkpoint.
           This method returns that node, to allow writing code like node.followUp(Node(...)).followUp(Node(...))."""
        if child._parent is not None:
            raise RuntimeError(f"Node '{child.handle}' already has a parent '{child._parent.handle}', so cannot make '{self.handle}' the parent.")

        self._children.append(child)
        child._parent = self
        return child

    def followUps(self, children: List["LineageNode"]):
        for child in children:
            self.followUp(child)

    @abstractmethod
    def duplicate(self) -> Self:
        """Get an object of the same subclass that shares all its fields with the current object, but with no child/parent relationships."""
        pass

    @abstractmethod
    def _run(self):
        """Do what this node should do to generate its output checkpoint."""
        pass

    def _getLineage(self) -> "Lineage":
        """Find the lineage by tracing back up to the starting node."""
        node = self
        while not isinstance(node, LineageRootNode):
            node = node._parent
        assert isinstance(node, LineageRootNode)
        return node._lineage

    def _buildHyperparameters(self) -> TaskHyperparameters:
        """
        Retrieve the node's hyperparameters while imputing the tokeniser and base model from the lineage this node belongs
        to, and imputing the checkpoint from the parent node.
        """
        basemodel_node = self
        while not(isinstance(basemodel_node, _AnchorNode) and basemodel_node.base_model is not None):
            basemodel_node = basemodel_node._parent

        tokeniser_node = self
        while not(isinstance(tokeniser_node, _AnchorNode) and tokeniser_node.tokeniser is not None):
            tokeniser_node = tokeniser_node._parent

        # Copy HPs because e.g. storing the tokeniser permanently in this node's HPs is a bad idea.
        hp = self._hp.copy()
        hp.MODEL_CONFIG_OR_CHECKPOINT = self._parent.out
        hp.archit_basemodel_class = basemodel_node.base_model
        hp.TOKENISER              = tokeniser_node.tokeniser

        # Identify the run by the full name of the lineage.
        lineage = self._getLineage()
        hp.SAVE_AS = lineage.name
        return hp

    def __iter__(self):
        yield self
        for node in self._children:
            yield from node.__iter__()


class _AnchorNode(LineageNode):
    def __init__(self, tokeniser: Optional[SerialisedTokeniser], base_model: Optional[Type[BaseModel]],
                 config_or_checkpoint: Optional[Union[Checkpoint, PretrainedConfig]]):
        super().__init__("", hp=None, out=config_or_checkpoint)
        self.tokeniser: Optional[SerialisedTokeniser] = tokeniser
        self.base_model: Optional[Type[BaseModel]]    = base_model


class LineageAnchorNode(_AnchorNode):
    """
    Resets the tokeniser and/or the base model architecture for all descendant nodes in the lineage.
    """
    def __init__(self, tokeniser: Optional[SerialisedTokeniser], base_model: Optional[Type[BaseModel]]):
        super().__init__(tokeniser=tokeniser, base_model=base_model, config_or_checkpoint=None)

    @property
    def out(self):
        return self._parent.out


class LineageRootNode(_AnchorNode):
    def __init__(self, starting_config_or_checkpoint: Union[Checkpoint, PretrainedConfig],
                 tokeniser: SerialisedTokeniser, base_model: Type[BaseModel]):
        super().__init__(tokeniser=tokeniser, base_model=base_model, config_or_checkpoint=starting_config_or_checkpoint)
        self._lineage: Lineage = None

    def _registerLineage(self, lineage: "Lineage"):
        self._lineage = lineage


class Lineage:
    """
    Ancestral tree of checkpoints. A basic tree may look something like:
    ```
                                                                  /---[tuning node]--> fine-tuned model
        [start]--> config --[training node]--> pretrained model --|---[tuning node]--> fine-tuned model
                                                                  \---[tuning node]--> fine-tuned model
    ```
    A lineage generally (unless in certain experimental setups) shares the same tokeniser and base model architecture
    across all its nodes.
    """

    def __init__(self, handle: str, name: str, root: LineageRootNode):
        self.handle = handle
        self.name   = name

        self._node_tree = root
        assert len(list(self.listHandles())) == len(set(self.listHandles())), "Lineage contains at least two nodes with the same handle."
        assert isinstance(self._node_tree, LineageRootNode)
        self._node_tree._registerLineage(self)

    def __iter__(self) -> Iterable[LineageNode]:
        yield from self._node_tree.__iter__()

    def run(self, node_handle: str):
        if not node_handle:
            raise ValueError(f"Cannot run empty node handle.")
        return self._get(node_handle)._run()

    def _get(self, node_handle: str) -> LineageNode:
        for node in self:
            if node.handle == node_handle:
                return node
        else:
            raise ValueError(f"Handle not in lineage: {node_handle}")

    def listHandles(self) -> List[str]:
        return [node.handle for node in self]


class LineageRegistry:
    """
    Collection of lineages, indexed by name.
    """

    def __init__(self):
        self._registry = dict()

    def add(self, lineage: Lineage):
        self._registry[lineage.handle] = lineage

    def get(self, handle: str) -> Lineage:
        try:
            return self._registry[handle]
        except:
            raise KeyError(f"No lineage with handle '{handle}' in registry with handles {self.listHandles()}")

    def listHandles(self) -> List[str]:
        return [l.handle for l in self]

    def __iter__(self) -> Iterable[Lineage]:
        for lineage in self._registry.values():
            yield lineage


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
        return TrainingNode(self.handle, self._hp, self._trainer, self._task)


class TuningNode(LineageNode):

    def __init__(self, handle: str, hp: TaskHyperparameters, meta: MetaHyperparameters, tuner: TaskTuner, task: Task, out: Optional[Checkpoint]=None):
        super().__init__(handle=handle, hp=hp, out=out)
        self._tuner = tuner
        self._task = task
        self._meta = meta

    def _run(self):
        result = self._tuner.tune(task=self._task, hp=self._buildHyperparameters(), meta=self._buildMetaHyperparameters())
        self._task.resetTemporaryFields()
        self._task.resetCaches()
        return result

    def duplicate(self) -> Self:
        return TuningNode(self.handle, self._hp, self._meta, self._tuner, self._task)

    def _buildMetaHyperparameters(self) -> MetaHyperparameters:
        """Adjust meta-hyperparameters so that the seed for taking grid samples is unique per lineage, per node."""
        lineage = self._getLineage()

        meta = self._meta.copy()
        meta.meta_seed += abs(hash(lineage.name)) + abs(hash(self.handle))
        return meta
