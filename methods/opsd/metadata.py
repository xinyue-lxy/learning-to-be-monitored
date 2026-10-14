"""Stable diagnostics for separately batched control/intervention workers."""


def initialize_contrast_diagnostics(extra_fields):
    # Controls apply the identity multiplier. Teacher postprocessing overwrites
    # these diagnostics for interventions; these values never change the loss.
    for name in ("mean", "max", "nonzero_fraction"):
        extra_fields.setdefault("opsd_teacher_contrast_" + name, 1.0)
