"""
Functional API called by the ReMoTO dashboard.
"""
from typing import TypeVar, Callable, Any

import sys
import json
import dacite

from lamoto.training.lineages import DictifyLineage, LineageRegistry

__all__ = ["api_GET_printJsonifiedRegistry", "api_POST_runLineageNode"]


Modifiables = TypeVar("Modifiables")

def api_GET_printJsonifiedRegistry(registry_factory: Callable[[Modifiables],LineageRegistry], modifiables_dataclass: type[Modifiables]):  # These are not command-line arguments because they are Python references, not just plain data.
    """
    Would be run in a raw call to Python, like
        cd YOUR_PROJECT_FOLDER
        YOUR_EXTRA_LINES
        echo 'YOUR_STRINGIFIED_JSON' | PYTHONPATH=. python -c "from YOUR_FACTORY_FILE import YOUR_FACTORY, YOUR_DATACLASS; from lamoto.training.integrations.remoto import *; api_GET_printJsonifiedRegistry(YOUR_FACTORY, YOUR_DATACLASS)" | tail -1
    """
    # Deserialise JSON on stdin
    registry = registryFromFactory(registry_factory, modifiables_dataclass, jsonFromStdin())

    # Serialise into Python dictionary, then into JSON, and then send to stdout
    dictionary = registryToDict(registry)
    print(json.dumps(dictionary))


def api_POST_runLineageNode(registry_factory: Callable[[Modifiables],LineageRegistry], modifiables_dataclass: type[Modifiables]):
    """
    Would be run from inside a Slurm script that looks like

        # SBATCH blabla
        # SBATCH blabla

        cd YOUR_PROJECT_FOLDER
        YOUR_EXTRA_LINES
        PYTHONPATH=. python -c "from YOUR_FACTORY_FILE import YOUR_FACTORY, YOUR_DATACLASS; from lamoto.training.integrations.remoto import *; api_POST_runLineageNode(YOUR_FACTORY, YOUR_DATACLASS)" $1 $2

    called like
        sbatch --input=/path/to/modifiables.json run.slurm 23 mlm
    """
    lineage_handle, node_handle = stringsFromArgs()
    registry = registryFromFactory(registry_factory, modifiables_dataclass, jsonFromStdin())
    registryRun(registry, lineage_handle, node_handle)


########################################################################################################################

def jsonFromStdin() -> dict:
    return json.load(sys.stdin)


def stringsFromArgs() -> list[str]:
    return sys.argv[1:]


def registryFromFactory(registry_factory: Callable[[Modifiables],LineageRegistry], modifiables_dataclass: type[Modifiables], modifiables_as_dict: dict[str,Any]) -> LineageRegistry:
    return registry_factory(dacite.from_dict(modifiables_dataclass, modifiables_as_dict))


def registryToDict(registry: LineageRegistry) -> dict:
    return {"lineages": [lineage.accept(DictifyLineage()) for lineage in registry]}


def registryRun(registry: LineageRegistry, lineage_handle: str, node_handle: str):
    return registry.get(lineage_handle).run(node_handle)
