from ._core import MetricRegistry, Metric
from .las import DependencyParsingMetrics
from .ppl import Perplexity
from .pppl import PseudoPerplexity
from .bpc import BitsPerCharacter
from .f1 import MacroF1, MacroRecall, MacroPrecision
from .qa import QA
from .aqa import AQA

METRICS = MetricRegistry()
METRICS.registerMetric("attachment", DependencyParsingMetrics)
METRICS.registerMetric("ppl", Perplexity)
METRICS.registerMetric("pppl", PseudoPerplexity)
METRICS.registerMetric("bpc", BitsPerCharacter)
METRICS.registerMetric("f1_macro", MacroF1)
METRICS.registerMetric("precision_macro", MacroPrecision)
METRICS.registerMetric("recall_macro", MacroRecall)
METRICS.registerMetric("qa", QA)
METRICS.registerMetric("aqa", AQA)
