"""CCI scoring engine — deterministic 10-step pipeline."""

from score.pipeline import CCIOutput, compute_cci

__all__ = ["compute_cci", "CCIOutput"]
