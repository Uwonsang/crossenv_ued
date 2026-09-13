"""Reproduce the earlier CEC versus CEC-IDAAC learning-curve comparison.

This entry point intentionally preserves the comparison version that preceded
the CEC-only figure. It shares CSV loading and plotting logic with
generalization_learning_curves.py, but defaults to both models and retains the
original per-NUM_ENVS output filenames.
"""

try:
    from .generalization_learning_curves import main
except ImportError:
    from generalization_learning_curves import main


if __name__ == "__main__":
    main(
        default_model_names=["CEC", "CEC_IDAAC"],
        output_tag=None,
        make_combined=False,
    )
