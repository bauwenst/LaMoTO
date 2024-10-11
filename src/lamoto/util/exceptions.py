from typing import Callable, TypeVar, Optional

T = TypeVar("T")


def tryExceptNone(executable: Callable[[],T]) -> Optional[T]:
    try:
        executable()
    except:
        return None
