"""MedSPOT: vendored core.compute_metrics under strict and ±14 px (medspot_metrics.both_rules), unchanged."""
from medspot_metrics import both_rules


def score(tasks, records):
    return both_rules(tasks, records)
