from ._core import MetricRegistry
from .las import DependencyParsingMetrics

METRICS = MetricRegistry()
METRICS.registerMetric("attachment", DependencyParsingMetrics)
