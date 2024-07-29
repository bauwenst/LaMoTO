from ._core import MetricRegistry
from .las import DependencyParsingMetrics
from .ppl import Perplexity
from .pppl import PseudoPerplexity
from .bpc import BitsPerCharacter

METRICS = MetricRegistry()
METRICS.registerMetric("attachment", DependencyParsingMetrics)
METRICS.registerMetric("ppl", Perplexity)
METRICS.registerMetric("pppl", PseudoPerplexity)
METRICS.registerMetric("bpc", BitsPerCharacter)
