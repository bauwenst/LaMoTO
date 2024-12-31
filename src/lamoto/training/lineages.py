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
from typing import Type, List, Iterable, Union, Optional, TypeVar, Iterator
from typing_extensions import Self
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from transformers import PretrainedConfig
Checkpoint = Union[str,Path]

from archit.instantiation.abstracts import BaseModel
from tktkt.util.strings import indent
from tktkt.interfaces.factories import TokeniserFactory
from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
SerialisedTokeniser = Union[str,TokeniserFactory[TokeniserWithFiniteTypeDomain]]

from .auxiliary.hyperparameters import TaskHyperparameters
from .training import TaskTrainer, Task
from .tuning import TaskTuner, MetaHyperparameters


LN = TypeVar("LN", bound="_LineageNode")
class _LineageNode(ABC):
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
        self._out   = out
        self._hp    = hp

        self._include_handle_in_checkpoint = False

        self._children: List[_LineageNode] = []
        self._parent: Optional[_LineageNode] = None

    @property
    def out(self):
        return self._out

    def followUp(self, child: LN) -> LN:
        """Add a node to the list of nodes that use the output of this node as their starting checkpoint.
           This method returns that node, to allow writing code like node.followUp(Node(...)).followUp(Node(...))."""
        if child._parent is not None:
            raise RuntimeError(f"Node '{child.handle}' already has a parent '{child._parent.handle}', so cannot make '{self.handle}' the parent.")

        self._children.append(child)
        child._parent = self
        return child

    def followUps(self, children: List["_LineageNode"]):
        for child in children:
            self.followUp(child)

    @abstractmethod
    def duplicate(self) -> Self:
        """Get an object of the same subclass that shares all its fields with the current object,
           but with no child/parent/lineage relationships."""
        pass

    def doIncludeHandleInOutput(self):
        """Mention the name of the node in output created by it."""
        self._include_handle_in_checkpoint = True

    @abstractmethod
    def _run(self):
        """Do what this node should do to generate its output checkpoint. (Protected method so that users building lineages don't have it suggested to them.)"""
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
        lineage = self._getLineage()
        if self._parent.out is None:
            raise RuntimeError(f"Node '{self.handle}' of lineage '{lineage.name}' is downstream of a node without a checkpoint. Run that node first.")

        basemodel_node = self
        while not(isinstance(basemodel_node, _AnchorNode) and basemodel_node.base_model is not None):
            basemodel_node = basemodel_node._parent
        assert basemodel_node.base_model is not None

        tokeniser_node = self
        while not(isinstance(tokeniser_node, _AnchorNode) and tokeniser_node.tokeniser is not None):
            tokeniser_node = tokeniser_node._parent
        assert tokeniser_node.base_model is not None

        # Copy HPs because e.g. storing the built tokeniser permanently in this node's HPs would be a memory leak in case we want to run multiple lineage nodes back-to-back in the same runtime session.
        hp = self._hp.copy()
        hp.MODEL_CONFIG_OR_CHECKPOINT = self._parent.out
        hp.archit_basemodel_class = basemodel_node.base_model
        hp.TOKENISER              = tokeniser_node.tokeniser
        hp.init_weights           = isinstance(hp.MODEL_CONFIG_OR_CHECKPOINT, (str,Path))  # In the event that you want to use a checkpoint's config only, pass in that config directly with AutoConfig.from_pretrained(chkpt).

        # Identify the run by the full name of the lineage and possibly the node.
        hp.SAVE_AS = lineage.name + ("_" + self.handle)*self._include_handle_in_checkpoint
        return hp

    def __iter__(self) -> Iterator["_LineageNode"]:
        yield self
        for node in self._children:
            yield from node.__iter__()

    def __repr__(self):
        s = (self.handle or "{anchor}") + ": " + self.__class__.__name__ + "(" + self._repr__args() + ")" + "\n"
        for child in self._children:
            s += indent(1, child.__repr__(), tab="|   ")
        return s

    def _repr__args(self) -> str:
        return ""


class _AnchorNode(_LineageNode, ABC):
    """
    Node from which each property (e.g. the tokeniser) that isn't None is copied by all descendants for which there is
    no anchor node between this and them with the same property not None.
    """
    def __init__(self, tokeniser: Optional[SerialisedTokeniser], base_model: Optional[Type[BaseModel]],
                 config_or_checkpoint: Optional[Union[Checkpoint, PretrainedConfig]]):
        super().__init__("", hp=None, out=config_or_checkpoint)
        self.tokeniser: Optional[SerialisedTokeniser] = tokeniser
        self.base_model: Optional[Type[BaseModel]]    = base_model

    def _run(self):
        raise NotImplementedError("Anchor nodes cannot be run.")

    def _buildHyperparameters(self) -> TaskHyperparameters:  # This also prevents root nodes from needing to access self._parent.out.
        raise NotImplementedError("Anchor nodes have no hyperparameters.")


class LineageAnchorNode(_AnchorNode):
    """
    Resets the tokeniser and/or the base model architecture for all descendant nodes in the lineage.
    """
    def __init__(self, tokeniser: Optional[SerialisedTokeniser]=None, base_model: Optional[Type[BaseModel]]=None):
        super().__init__(tokeniser=tokeniser, base_model=base_model, config_or_checkpoint=None)

    @property
    def out(self):
        return self._parent.out

    def duplicate(self) -> Self:
        return LineageAnchorNode(tokeniser=self.tokeniser, base_model=self.base_model)

    def _repr__args(self) -> str:
        fields_reset = []
        if self.tokeniser:
            fields_reset.append("tokeniser")
        if self.base_model:
            fields_reset.append("basemodel")

        return ",".join(fields_reset)


class LineageRootNode(_AnchorNode):
    def __init__(self, starting_config_or_checkpoint: Union[Checkpoint, PretrainedConfig],
                 tokeniser: SerialisedTokeniser, base_model: Type[BaseModel]):
        super().__init__(tokeniser=tokeniser, base_model=base_model, config_or_checkpoint=starting_config_or_checkpoint)
        self._lineage: Lineage = None

    def _registerLineage(self, lineage: "Lineage"):
        self._lineage = lineage

    def duplicate(self) -> Self:
        return LineageRootNode(starting_config_or_checkpoint=self.out,
                               tokeniser=self.tokeniser, base_model=self.base_model)


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
        assert isinstance(self._node_tree, LineageRootNode)

        duplicate_handles = [k for k,v in Counter(self.listHandles()).items() if v > 1]
        if duplicate_handles:
            raise ValueError(f"Lineage contains at least two nodes with the same handle: {duplicate_handles}")

        self._node_tree._registerLineage(self)

    def __iter__(self) -> Iterator[_LineageNode]:
        yield from filter(lambda node: node.handle != "", self._node_tree.__iter__())

    def run(self, node_handle: str):
        return self._get(node_handle)._run()

    def _get(self, node_handle: str) -> _LineageNode:
        if not node_handle:
            raise ValueError(f"Cannot retrieve empty node handle.")

        for node in self:
            if node.handle == node_handle:
                return node
        else:
            raise ValueError(f"Handle not in lineage: {node_handle}")

    def listHandles(self) -> List[str]:
        return [node.handle for node in self if node.handle]  # Don't include the anchors.

    def __repr__(self):
        s = "Lineage(" + self.name + ")\n"
        s += indent(1, self._node_tree.__repr__(), tab="|   ")
        return s


class LineageRegistry:
    """
    Collection of lineages, indexed by name.
    """

    def __init__(self):
        self._registry = dict()

    def add(self, lineage: Lineage):
        if lineage.handle in self._registry:
            raise ValueError(f"Lineage handle already in registry: {lineage.handle}")
        self._registry[lineage.handle] = lineage

    def get(self, handle: str) -> Lineage:
        try:
            return self._registry[handle]
        except:
            raise KeyError(f"No lineage with handle '{handle}' in registry with handles {self.listHandles()}")

    def listHandles(self) -> List[str]:
        return [l.handle for l in self]

    def listNames(self) -> List[str]:
        return [l.name for l in self]

    def __iter__(self) -> Iterable[Lineage]:
        for lineage in self._registry.values():
            yield lineage


class TrainingNode(_LineageNode):

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

    def _repr__args(self) -> str:
        return self._task.task_name


class TuningNode(_LineageNode):

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

    def _repr__args(self) -> str:
        return self._task.__class__.__name__

    def _buildMetaHyperparameters(self) -> MetaHyperparameters:
        """Adjust meta-hyperparameters so that the seed for taking grid samples is unique per lineage, per node."""
        lineage = self._getLineage()

        meta = self._meta.copy()
        meta.meta_seed += abs(hash(lineage.name)) + abs(hash(self.handle))
        return meta
