from typing import Callable, TypeVar, Optional

T = TypeVar("T")


def tryExceptNone(executable: Callable[[],T]) -> Optional[T]:
    try:
        return executable()
    except:
        return None
