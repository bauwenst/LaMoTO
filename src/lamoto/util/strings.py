from typing import Any


def getSubstringAfterLastSlash(s: str):
    return s[s.rfind("/")+1:]


def prefixKeys(data: dict[str,Any], prefix: str) -> dict[str,Any]:
    data = data.copy()
    for key in list(data.keys()):
        if not key.startswith(f"{prefix}_"):
            data[f"{prefix}_{key}"] = data.pop(key)
    return data
