"""Offline recall-bound arithmetic; no inference, label edits or policy changes.

Callers must retain their frozen task gate and declared support before using
these helpers. Inputs are full saved per-instance metrics, not aggregate TPR.
Only Python's standard library is required. See docs/METRICS.md.
"""
import math
from statistics import fmean

FIELDS = ("lower", "upper", "TNR_X1", "lower_score", "upper_score")
CONTEXT = ("monitor", "seed", "evaluation_seed", "role", "model", "task")


def _probability(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not 0 <= value <= 1 or (positive and value == 0):
        raise ValueError(f"{name} must be in {'(0, 1]' if positive else '[0, 1]'}")
    return value


def recall_bounds(r_effect, p_z1_x1_y1):
    """Return (L, U); r_effect is (p1-p0)/p1, NOT the raw p1-p0."""
    q = _probability(r_effect, "r_effect", positive=True)
    z = _probability(p_z1_x1_y1, "p_z1_x1_y1")
    return max(0.0, (z + q - 1.0) / q), min(1.0, z / q)


def instance_lower_metrics(record):
    """Preserve existing product-estimable support; return None for saved NA.

    Reject incomplete components, failed gates, or disagreement with the saved
    upper estimator. No missing/invalid verdict is ever converted to Z=0.
    """
    if "modified_gmean2_i" not in record:
        raise ValueError("Full per-instance metric required: missing modified_gmean2_i")
    if record["modified_gmean2_i"] is None:
        return None
    required = ("r_effect", "p_z1_x1_y1", "TPR_mc_i", "TNR_X1_i", "metric_valid",
                "effect_eligible", "tnr_estimable", "n_X1_Y0_valid_monitor_task_rollouts")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"Full per-instance components required; missing: {missing}")
    if not all(record[name] is True for name in ("metric_valid", "effect_eligible", "tnr_estimable")):
        raise ValueError("Saved non-NA product has invalid support flags")
    negatives = record["n_X1_Y0_valid_monitor_task_rollouts"]
    if type(negatives) is not int or negatives < 2:
        raise ValueError("At least two distinct valid X1,Y0 task rollouts are required")
    lower, upper = recall_bounds(record["r_effect"], record["p_z1_x1_y1"])
    tnr = _probability(record["TNR_X1_i"], "TNR_X1_i")
    old_upper = _probability(record["TPR_mc_i"], "TPR_mc_i")
    old_score = _probability(record["modified_gmean2_i"], "modified_gmean2_i")
    if not (math.isclose(upper, old_upper, rel_tol=0, abs_tol=1e-10) and
            math.isclose(upper * tnr, old_score, rel_tol=0, abs_tol=1e-10)):
        raise ValueError("Saved upper bound/product does not match the supplied components")
    return dict(lower=lower, upper=upper, TNR_X1=tnr,
                lower_score=lower * tnr, upper_score=upper * tnr)


def _mean(values):
    return {field: fmean(value[field] for value in values) for field in FIELDS}


def _context(rows, keys):
    context = {}
    for key in keys:
        if any(key in row for row in rows):
            if not all(key in row and row[key] == rows[0].get(key) for row in rows):
                raise ValueError(f"Do not pool different or missing {key} identities")
            context[key] = rows[0][key]
    return context


def summarize_task(records, *, task_reportable):
    """Mean instance quantities on caller-selected support and frozen task gate.

    task_reportable must come from the existing task report (values != None),
    not from whether this new lower-bound calculation happens to be positive.
    For common support, prefilter both roles to the same declared ID set.
    """
    if type(task_reportable) is not bool:
        raise ValueError("Explicit frozen task_reportable boolean required")
    rows = list(records)
    context = _context(rows, CONTEXT)
    if any("task_id" not in row for row in rows):
        raise ValueError("Saved task_id required for duplicate/support checks")
    ids = [row["task_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate instance ID within one task/role")
    pairs = [(row["task_id"], instance_lower_metrics(row)) for row in rows]
    retained = [(iid, value) for iid, value in pairs if value is not None]
    if task_reportable and not retained:
        raise ValueError("Reportable task has no estimable instances")
    return dict(values=_mean([value for _, value in retained]) if task_reportable else None,
                native_or_selected_ids=[iid for iid, _ in retained],
                estimable=len(retained), input_instances=len(rows), context=context,
                task_reportable=task_reportable)

