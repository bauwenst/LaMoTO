from ..glue._general import GLUETask, CompareSentencesGLUETask, ClassifySentenceGLUETask


class SuperGLUETask(GLUETask):
    BASE_REPO = "aps/super_glue"


class CompareSentencesSuperGLUETask(CompareSentencesGLUETask):
    BASE_REPO = "aps/super_glue"


class ClassifySentenceSuperGLUETask(ClassifySentenceGLUETask):
    BASE_REPO = "aps/super_glue"
