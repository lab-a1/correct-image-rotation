from collections import defaultdict


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"value": 0, "count": 0, "avg": 0})

    def update(self, metric_name, value):
        metric = self.metrics[metric_name]

        metric["value"] += value
        metric["count"] += 1
        metric["avg"] = metric["value"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                f"{metric_name}: {metric['avg']:.{self.float_precision}f}"
                for (metric_name, metric) in self.metrics.items()
            ]
        )
