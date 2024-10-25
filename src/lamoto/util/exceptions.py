from typing import Callable, TypeVar, Optional

T = TypeVar("T")


def tryExceptNone(executable: Callable[[],T]) -> Optional[T]:
    try:
        return executable()
    except:
        return None


class ImpossibleBranchError(RuntimeError):
    def __init__(self):
        super().__init__("Reached part of the code that should be unreachable.")
