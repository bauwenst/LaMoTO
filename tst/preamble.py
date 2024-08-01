from pathlib import Path

PATH_TST = Path(__file__).parent
PATH_ROOT = PATH_TST.parent
PATH_DATA = PATH_ROOT / "data"
PATH_DATA_OUT = PATH_DATA / "out"
PATH_DATA_OUT.mkdir(exist_ok=True, parents=True)

from tktkt.files.paths import setTkTkToutputRoot
setTkTkToutputRoot(PATH_DATA_OUT)
