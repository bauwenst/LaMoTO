from ..glue._general import CompareSentencesGLUETask, GLUETask


class SuperGLUETask(GLUETask):
    BASE_REPO = "aps/super_glue"


class CompareSentencesSuperGLUETask(CompareSentencesGLUETask):
    BASE_REPO = "aps/super_glue"
