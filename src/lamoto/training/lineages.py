"""
You can use LaMoTO for one-off model training. For research, however, we often compare models. This is done by first
pretraining given configurations, and then taking each checkpoint and fine-tuning them on multiple tasks.

Although you could see all of these experiments as entirely separate, it makes more sense to speak in terms of an
"ancestry" or a "lineage" of models. Within one lineage, the model backbone and the tokeniser always stay the same, but
the weights change, the hyperparameters change, and the head changes.
When you want to set up training experiments from the command line, rather than having to declare the tokeniser and
checkpoint on the command line, it's nicer to be able to just request to "train node X in lineage A" and have the backend
load checkpoints and tokenisers for you.

If I'd redesign this again, I would perhaps have nodes be checkpoints rather than the arcs between them being checkpoints,
and I would also perhaps let Lineage and LineageRootNode be the same class, rather than one wrapping the other.
The only real benefit you get from this wrapping is that you don't get a ".run()" method suggested while building a node tree.

Typically, when using lineages, a script defining your language modelling experiments will look like this:
    1. Define all your root nodes;
    2. Define a placeholder followed by your pretraining node;
    3. Define your task nodes on the placeholder;
    4. Build a registry from your root nodes followed by that placeholder;
    5. Define where to find existing outputs of some nodes (if applicable).
Then after this, you create a script that parses command-line arguments and uses the one-liner
    REGISTRYNAME.get(LINEAGENAME).run(NODENAME)
to run an experiment.
"""
from typing import Type, List, Iterable, Union, Optional, TypeVar, Iterator, Tuple, Callable, Generic
from typing_extensions import Self
from pathlib import Path
from collections import Counter
from abc import ABC, abstractmethod
from transformers import PretrainedConfig
from dataclasses import asdict

import itertools

from archit.instantiation.abstracts import BaseModel
from tktkt.interfaces import TokeniserFactory, TokeniserWithVocabulary
from tktkt.util.strings import indent
from tktkt.util.iterables import filterOptionals
SerialisedTokeniser = Union[str,TokeniserFactory[TokeniserWithVocabulary]]

from .auxiliary.hyperparameters import TaskHyperparameters
from .training import TaskTrainer, Task
from .tuning import TaskTuner, MetaHyperparameters
from ..util.exceptions import tryExceptNone
from ..util.visuals import warn


class ConfigFactory(ABC):
    @abstractmethod
    def buildConfig(self, base_model: Type[BaseModel], serialised_tokeniser: SerialisedTokeniser) -> PretrainedConfig:
        pass

Checkpoint = Union[str,Path]
Config     = Union[PretrainedConfig,ConfigFactory]
NodeOutput = Union[Config,Checkpoint]

T  = TypeVar("T")
LN = TypeVar("LN", bound="_LineageNode")
class _LineageNode(ABC):
    """
    Tree node representing one operation that advances a lineage.
    """

    def __init__(self, handle: str):
        self.handle = handle

        self._children: List[_LineageNode] = []
        self._parent: Optional[_LineageNode] = None

    def next(self, child: LN) -> LN:
        """Add a node to the list of nodes that use the output of this node as their starting checkpoint.
           This method returns that node, to allow writing code like node.next(Node(...)).next(Node(...))."""
        if child._parent is not None:
            raise RuntimeError(f"Node '{child.handle}' already has a parent '{child._parent.handle}', so cannot make node '{self.handle}' the parent.")
        elif any(ancestor in child for ancestor in self._getAncestors()):
            if self == child:
                raise RuntimeError(f"Tried to make node '{child.handle}' a child of itself.")
            elif self in child:
                raise RuntimeError(f"Tried to make node '{child.handle}' a child of node '{self.handle}' even though the latter is one of its descendants.")
            else:
                raise RuntimeError(f"Failed to concatenate node trees: one of the ancestors of node '{self.handle}' is a descendant of '{child.handle}', causing a loop in the tree.")

        self._children.append(child)
        child._parent = self
        return child

    @property
    @abstractmethod
    def _out(self) -> NodeOutput:
        pass

    @abstractmethod
    def accept(self, visitor: "_LineageNodeVisitor[T]") -> T:
        pass

    def nextInParalllel(self, children: List["_LineageNode"]):
        for child in children:
            self.next(child)

    @abstractmethod
    def duplicate(self) -> Self:
        """Get an object of the same subclass that shares all its fields with the current object,
           but with no child/parent/lineage relationships."""
        pass

    def duplicateTree(self) -> Self:
        """Same as duplicate() except you append duplicates of all the descendants too."""
        this_copy = self.duplicate()
        for child in self._children:
            this_copy.next(child.duplicateTree())
        return this_copy

    # Walking up the tree

    def _getAncestors(self) -> List["_LineageNode"]:
        """Trace back the parent relationship until the top of the tree."""
        nodes = [self]
        while nodes[-1]._parent is not None:  # When this condition triggers, nodes[-1] is the top of the tree.
            nodes.append(nodes[-1]._parent)
        return nodes

    def _getRoot(self) -> Optional["LineageRootNode"]:
        """Returns the top of the tree if it is a root node, else None."""
        top = self._getAncestors()[-1]
        return top if isinstance(top, LineageRootNode) else None

    def _getLineage(self) -> Optional["Lineage"]:
        root = self._getRoot()
        return None if root is None else root._lineage

    def _getTokeniserFactory(self) -> SerialisedTokeniser:
        return self._parent._getTokeniserFactory()

    def _getBaseModelClass(self) -> type[BaseModel]:
        return self._parent._getBaseModelClass()

    # Walking down the tree

    def __iter__(self) -> Iterator["_LineageNode"]:
        yield self
        for node in self._children:
            yield from node.__iter__()

    def __repr__(self):
        return self.accept(StringifyLineage())


class _AnchorNode(_LineageNode, ABC):
    """
    Node from which each property (e.g. the tokeniser) that isn't None is copied by all descendants for which there is
    no anchor node between this and them with the same property not None.
    """
    def __init__(self, base_model: Optional[Type[BaseModel]], tokeniser: Optional[SerialisedTokeniser]):
        super().__init__(handle="")
        self._base_model: Optional[Type[BaseModel]]    = base_model
        self._tokeniser: Optional[SerialisedTokeniser] = tokeniser

    def _getTokeniserFactory(self) -> SerialisedTokeniser:
        return super()._getTokeniserFactory() if self._tokeniser is None else self._tokeniser

    def _getBaseModelClass(self) -> type[BaseModel]:
        return super()._getBaseModelClass() if self._base_model is None else self._base_model


class LineageAnchorNode(_AnchorNode):
    """
    Resets the tokeniser and/or the base model architecture for all descendant nodes in the lineage.
    """
    @property
    def _out(self):
        return self._parent._out

    def duplicate(self) -> Self:
        return LineageAnchorNode(tokeniser=self._tokeniser, base_model=self._base_model)

    def accept(self, visitor: "_LineageNodeVisitor[T]") -> T:
        return visitor.visitAnchorNode(self)


class LineageRootNode(_AnchorNode):
    def __init__(self, name: str, starting_config_or_checkpoint: NodeOutput,
                 base_model: Type[BaseModel], tokeniser: SerialisedTokeniser):
        super().__init__(tokeniser=tokeniser, base_model=base_model)
        self._config_or_checkpoint = starting_config_or_checkpoint
        self._lineage: Lineage = None
        self.name = name

    @property
    def _out(self):
        return self._config_or_checkpoint

    def _registerLineage(self, lineage: "Lineage"):
        self._lineage = lineage

    def duplicate(self) -> Self:
        return LineageRootNode(name=self.name, starting_config_or_checkpoint=self._config_or_checkpoint,
                               tokeniser=self._tokeniser, base_model=self._base_model)

    def accept(self, visitor: "_LineageNodeVisitor[T]") -> T:
        return visitor.visitRootNode(self)


class LineagePlaceholderNode(_LineageNode):
    """
    Multiple lineage trees built in parallel. When you add a node to one tree, it is added to all other trees in the
    same location.
    """

    def __init__(self):
        super().__init__(handle="")

    def duplicate(self) -> Self:
        return LineagePlaceholderNode()

    def after(self, parent: _LineageNode) -> _LineageNode:
        """
        Concatenates a copy of this node tree to the given node in-place, but without the placeholder node.
        a.next(b) is roughly equivalent to b.after(a) except the placeholder node itself disappears.
        Returns the given node, which now has the same descendants as this placeholder.
        """
        for child_tree in self.duplicateTree()._children:
            child_tree._parent = None
            parent.next(child_tree)
        return parent

    def afterMany(self, parents: List[_LineageNode]):
        """Same as after() except in bulk."""
        for parent in parents:
            self.after(parent)

    def buildRegistry(self, starting_nodes: List[_LineageNode], quiet: bool=True) -> "LineageRegistry":
        """
        Append this node tree to each of the given nodes, turn the resulting trees into lineages stored in a registry,
        and print these lineages.
        """
        self.afterMany(starting_nodes)

        registry = LineageRegistry()
        for i,node in enumerate(starting_nodes):
            lineage = Lineage(handle=f"{i+1}", root=node._getRoot())
            registry.add(lineage)
            if not quiet:
                print(lineage)
        return registry

    @property
    def _out(self) -> NodeOutput:
        raise RuntimeError("Placeholder nodes cannot produce anything.")

    def _getTokeniserFactory(self) -> SerialisedTokeniser:
        raise RuntimeError("Placeholder nodes have no tokeniser.")

    def _getBaseModelClass(self) -> type[BaseModel]:
        raise RuntimeError("Placeholder nodes have no base model.")

    def _getLineage(self) -> Optional["Lineage"]:
        raise RuntimeError("Placeholder nodes are lineage-agnostic.")

    def accept(self, visitor: "_LineageNodeVisitor[T]") -> T:
        raise RuntimeError("Placeholder nodes cannot be visited.")


class Lineage:
    """
    Ancestral tree of checkpoints. A basic tree may look something like:
    ```
                                                                  /---[tuning node]--> fine-tuned model
         [root]--> config --[training node]--> pretrained model --|---[tuning node]--> fine-tuned model
                                                                  \---[tuning node]--> fine-tuned model
    ```
    A lineage generally (unless in certain experimental setups) shares the same tokeniser and base model architecture
    across all its nodes.
    """

    def __init__(self, handle: str, root: LineageRootNode):
        # Safety checks
        assert isinstance(root, LineageRootNode), "The top of a lineage must be a root node."
        assert all(not isinstance(node, (LineageRootNode, LineagePlaceholderNode)) or i == 0 for i,node in enumerate(root)), "The lineage cannot contain placeholders or more than one root."

        duplicate_handles = [k for k,v in Counter(filter(lambda handle: handle != "", map(lambda node: node.handle, root))).items() if v > 1]
        if duplicate_handles:
            raise ValueError(f"Lineage contains at least two nodes with the same handle: {duplicate_handles}")

        # Store arguments and set bidirectional association
        self.handle = handle
        self._node_tree = root
        self._node_tree._registerLineage(self)

    def __iter__(self) -> Iterator[_LineageNode]:
        yield from filter(lambda node: node.handle != "", self._node_tree.__iter__())

    def run(self, node_handle: str):
        return self._get(node_handle)._run()

    def out(self, node_handle: str, node_output: NodeOutput):
        return self._get(node_handle).out(node_output)

    def _get(self, node_handle: str) -> "_RunnableNode":
        if not node_handle:
            raise ValueError(f"Cannot retrieve empty node handle.")

        for node in self:
            if node.handle == node_handle:
                if not isinstance(node, _RunnableNode):
                    raise RuntimeError("Lineages only allow looking up runnable nodes.")
                return node
        else:
            raise ValueError(f"Handle not in lineage: {node_handle}")

    def listHandles(self) -> List[str]:
        return [node.handle for node in self]

    def __repr__(self):
        return self.accept(StringifyLineage())

    @property
    def name(self):
        return self._node_tree.name

    def accept(self, visitor: "_LineageNodeVisitor[T]") -> T:
        return visitor.visitLineage(self)


class LineageRegistry:
    """
    Collection of lineages, indexed by name.
    """

    def __init__(self):
        self._registry = dict()

    def add(self, lineage: Lineage):
        if lineage.handle in self._registry:
            raise ValueError(f"Lineage handle already in registry: {lineage.handle}")
        if lineage.name in {l.name for l in self}:
            raise ValueError(f"Lineage name already in registry (under handle '{lineage.handle}'): {lineage.name}")
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

    def mapNodeHandles(self, f: Callable[[str,str],str]) -> Iterable[List[str]]:
        """
        Attempts to align the nodes of all the lineages in the registry and then for each stack of aligned nodes returns
        the list of results for applying f(lineage handle, node handle) to the nodes in that stack.
        """
        pair_generators: List[Iterator[Tuple[Lineage,_LineageNode]]] = [zip(itertools.repeat(l), reversed(list(l))) for l in self]
        results = []
        while True:
            pairs = [tryExceptNone(lambda: next(g)) for g in pair_generators]
            new_results = [f(l.handle,n.handle) for l,n in filterOptionals(pairs)]
            if not new_results:
                break
            results.append(new_results)

        return reversed(results)


class _RunnableNode(_LineageNode):

    def __init__(self, handle: str, hp: TaskHyperparameters, out: Optional[NodeOutput]=None):
        """
        :param hp: Some of the given hyperparameter values will always be ignored, e.g. the base model type and the tokeniser (because
                   those come from the lineage), and for finetuning, every hyperparameter in the grid.
        :param out: Checkpoint identifier indicating the RESULT of running this lineage node.
                    You leave this parameter None when you run your experiment, and then fill it in when you want to
                    do follow-up experiments.
        """
        super().__init__(handle=handle)
        self._hp           = hp
        self._include_handle_in_checkpoint = False
        self._out_as_field = out

    @property
    def _out(self) -> NodeOutput:
        return self._out_as_field

    def out(self, node_output: NodeOutput) -> Self:
        """Sets the output of this node in-place. (If a file path, we warn about its inexistence, but don't error on it.)"""
        if isinstance(node_output, (str, Path)) and not Path(node_output).exists():
            warn(f"Output of lineage node does not exist yet: {Path(node_output)}")
        self._out_as_field = node_output
        return self

    def doIncludeHandleInOutput(self):
        """Mention the name of the node in output created by it."""
        self._include_handle_in_checkpoint = True

    @abstractmethod
    def _run(self):
        """Do what this node should do to generate its output checkpoint. (Protected method so that users building lineages don't have it suggested to them.)"""
        pass

    def _buildHyperparameters(self) -> TaskHyperparameters:
        """
        Retrieve the node's hyperparameters while imputing the tokeniser and base model from the lineage this node belongs
        to, and imputing the checkpoint from the parent node.
        """
        lineage = self._getLineage()
        assert lineage is not None
        if self._parent._out is None:
            raise RuntimeError(f"Node '{self.handle}' of lineage '{lineage.name}' is downstream of a node without a checkpoint. Run that node first.")

        parent_out = self._parent._out
        base_model = self._getBaseModelClass()
        tokeniser  = self._getTokeniserFactory()

        # Copy HPs because e.g. storing the built tokeniser permanently in this node's HPs would be a memory leak in case we want to run multiple lineage nodes back-to-back in the same runtime session.
        hp = self._hp.copy()
        hp.model_config_or_checkpoint = parent_out if not isinstance(parent_out, ConfigFactory) else parent_out.buildConfig(serialised_tokeniser=tokeniser, base_model=base_model)
        hp.archit_basemodel_class = base_model
        hp.tokeniser              = tokeniser
        hp.init_weights           = isinstance(hp.model_config_or_checkpoint, (str, Path))  # In the event that you want to use a checkpoint's config only, pass in that config directly with AutoConfig.from_pretrained(chkpt).

        # Identify the run by the full name of the lineage and possibly the node.
        hp.save_as = lineage.name + ("_" + self.handle)*self._include_handle_in_checkpoint
        return hp


class TrainingNode(_RunnableNode):

    def __init__(self, handle: str, hp: TaskHyperparameters, trainer: TaskTrainer, task: Task, out: Optional[Checkpoint]=None):
        super().__init__(handle=handle, hp=hp, out=out)
        self._trainer = trainer
        self._task = task

    def _run(self):
        result = self._trainer.train(task=self._task, hyperparameters=self._buildHyperparameters())
        self._task.resetTemporaryFields()
        # self._task.resetCaches()
        return result

    def duplicate(self) -> Self:
        return TrainingNode(self.handle, self._hp, self._trainer, self._task)

    def accept(self, visitor: "_LineageNodeVisitor[T]") -> T:
        return visitor.visitTrainingNode(self)


class TuningNode(_RunnableNode):

    def __init__(self, handle: str, hp: TaskHyperparameters, meta: MetaHyperparameters, tuner: TaskTuner, task: Task, out: Optional[Checkpoint]=None):
        super().__init__(handle=handle, hp=hp, out=out)
        self._tuner = tuner
        self._task = task
        self._meta = meta

    def _run(self):
        result = self._tuner.tune(task=self._task, hp=self._buildHyperparameters(), meta=self._buildMetaHyperparameters())
        self._task.resetTemporaryFields()
        # self._task.resetCaches()
        return result

    def duplicate(self) -> Self:
        return TuningNode(self.handle, self._hp, self._meta, self._tuner, self._task)

    def _buildMetaHyperparameters(self) -> MetaHyperparameters:
        """Adjust meta-hyperparameters so that the seed for taking grid samples is unique per lineage, per node."""
        lineage = self._getLineage()
        assert lineage is not None

        meta = self._meta.copy()
        meta.meta_seed += abs(hash(lineage.name)) + abs(hash(self.handle))
        return meta

    def accept(self, visitor: "_LineageNodeVisitor[T]") -> T:
        return visitor.visitTuningNode(self)


########################################################################################################################


class _LineageNodeVisitor(ABC, Generic[T]):
    @abstractmethod
    def visitRootNode(self, node: LineageRootNode) -> T:
        pass

    @abstractmethod
    def visitAnchorNode(self, node: LineageAnchorNode) -> T:
        pass

    @abstractmethod
    def visitTrainingNode(self, node: TrainingNode) -> T:
        pass

    @abstractmethod
    def visitTuningNode(self, node: TuningNode) -> T:
        pass

    @abstractmethod
    def visitLineage(self, lineage: Lineage) -> T:
        pass


class StringifyLineage(_LineageNodeVisitor[str]):

    def stringifyNode(self, node: _LineageNode, extras: str):
        s = (node.handle or "{anchor}") + ": " + node.__class__.__name__ + "(" + extras + ")" + "\n"
        for child in node._children:
            s += indent(1, child.accept(self), tab="|   ")
        return s

    def visitRootNode(self, node: LineageRootNode):
        return self.stringifyNode(node, extras="")

    def visitAnchorNode(self, node: LineageAnchorNode):
        fields_reset = []
        if node._base_model:
            fields_reset.append("basemodel")
        if node._tokeniser:
            fields_reset.append("tokeniser")

        return self.stringifyNode(node, extras=",".join(fields_reset))

    def visitTrainingNode(self, node: TrainingNode):
        return self.stringifyNode(node, extras=node._task.task_name)

    def visitTuningNode(self, node: TuningNode):
        return self.stringifyNode(node, extras=node._task.task_name)

    def visitLineage(self, lineage: Lineage) -> T:
        s = f"Lineage {lineage.handle}: \"{lineage.name}\"\n"
        s += indent(1, lineage._node_tree.accept(self), tab="|   ")
        return s


class DictifyLineage(_LineageNodeVisitor[dict]):

    def jsonifyHyperparameters(self, hp: TaskHyperparameters) -> dict:
        d = hp.to_dict()
        d.pop("archit_basemodel_class")
        d.pop("custom_hf_class")
        d.pop("model_config_or_checkpoint")
        d.pop("tokeniser")
        d.pop("init_weights")
        d.pop("_hp_class")
        d.pop("save_as")
        return d

    def recurse(self, node: _LineageNode, node_properties: dict) -> dict:
        return {"type": node.__class__.__name__, "output": node._out is not None} | {"properties": node_properties} | {"children": [child.accept(self) for child in node._children]}

    def visitRootNode(self, node: LineageRootNode):
        d = dict()

        # Deserialise config/checkpoint
        output = node._config_or_checkpoint
        if isinstance(output, Path):
            output = output.as_posix()
        elif isinstance(output, ConfigFactory):
            output = output.buildConfig(base_model=node._getBaseModelClass(), serialised_tokeniser=node._getTokeniserFactory())

        if isinstance(output, str):
            d["checkpoint"] = output
        elif isinstance(output, PretrainedConfig):
            # d["config"] = output.to_dict()
            d["config"] = output.__class__.__name__  # The entire config is uselessly big... better just send its class.
        else:
            raise NotImplementedError()

        d["tokeniser"] = node._getTokeniserFactory().buildTokeniser().getName()
        d["basemodel"] = node._getBaseModelClass().__name__
        return self.recurse(node, d)

    def visitAnchorNode(self, node: LineageAnchorNode):
        d = dict()
        if node._tokeniser is not None:
            d["tokeniser"] = node._getTokeniserFactory().buildTokeniser().getName()
        if node._base_model is not None:
            d["basemodel"] = node._getBaseModelClass().__name__

        return self.recurse(node, d)

    def visitTrainingNode(self, node: TrainingNode):
        d = dict()
        d["task"] = node._task.task_name
        if node._trainer._model_augmentation:
            d["augmentation"] = node._trainer._model_augmentation.name()
        d["hp"]   = self.jsonifyHyperparameters(node._hp)
        return self.recurse(node, d)

    def visitTuningNode(self, node: TuningNode):
        d = dict()
        d["task"] = node._task.task_name
        if node._tuner._trainer._model_augmentation:
            d["augmentation"] = node._tuner._trainer._model_augmentation.name()
        d["meta"] = asdict(node._meta)
        d["grids"] = {
            "guaranteed": {k: v for k,v in asdict(node._tuner._guaranteed     ).items() if v is not None} if node._tuner._guaranteed      is not None else None,
            "sampled":    {k: v for k,v in asdict(node._tuner._sampling_domain).items() if v is not None} if node._tuner._sampling_domain is not None else None,
        }
        d["hp"]   = self.jsonifyHyperparameters(node._hp)
        return self.recurse(node, d)

    def visitLineage(self, lineage: Lineage) -> T:
        return {
            "handle": lineage.handle,
            "name": lineage.name,
            "nodes": lineage._node_tree.accept(self)
        }
