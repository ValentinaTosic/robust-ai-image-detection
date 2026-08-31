"""Construct leave-one-generator-out (LOGO) dataset manifests.

Builds manifest variants for evaluating generalization to AI generators
that are completely unseen during training and model selection.
Held-out generator images are removed from the training and validation splits and
pooled into a dedicated test split alongside the real test images.
"""

import pandas as pd


def list_ai_generators(manifest: pd.DataFrame) -> list[str]:
    """Return the sorted list of AI generator names present in a manifest.

    Args:
        manifest: Dataset manifest containing "label" and "generator" columns.

    Returns:
        Sorted list of AI generator names, excluding real images.
    """
    return sorted(manifest.loc[manifest["label"] == "ai", "generator"].unique())


def build_logo_manifest(manifest: pd.DataFrame, unseen_generators: str | list[str]) -> pd.DataFrame:
    """Build a manifest with one or more AI generators fully held out.

    All AI images from the specified unseen generator(s), regardless of their
    original split, are assigned to the test split. AI test images from all
    other generators are removed so that the resulting test set contains only
    real images and the held-out generator(s).

    A single generator represents the standard LOGO evaluation. Multiple
    generators can be held out for targeted experiments, such as testing
    whether two architecturally related generators can generalize to each
    other.

    Real images and the train/validation AI images from all other generators
    remain unchanged. Therefore, the unseen generator(s) cannot influence
    training, early stopping, or model-selection decisions.

    Args:
        manifest: Full dataset manifest containing at least "label", "generator", and "split" columns.
        unseen_generators: AI generator name to hold out, or a list of
            generator names to hold out together. Each name must be present
            in the manifest for AI-labeled rows.

    Returns:
        A new manifest DataFrame. The input manifest is not modified. AI rows
        belonging to the unseen generator(s) are assigned to the test split,
        while other generators' original test-split AI rows are removed.

    Raises:
        ValueError: If any specified unseen generator is not present among
            the AI generators in the manifest.
    """
    if isinstance(unseen_generators, str):
        unseen_generators = [unseen_generators]

    available_generators: list[str] = list_ai_generators(manifest)
    unknown: list[str] = [g for g in unseen_generators if g not in available_generators]
    if unknown:
        raise ValueError(f"Unknown unseen_generators {unknown!r}; expected values from {available_generators}.")

    logo_manifest: pd.DataFrame = manifest.copy()

    is_unseen_ai: pd.Series = (logo_manifest["label"] == "ai") & (logo_manifest["generator"].isin(unseen_generators))
    is_other_generator_test_ai: pd.Series = (
        (logo_manifest["label"] == "ai")
        & (~logo_manifest["generator"].isin(unseen_generators))
        & (logo_manifest["split"] == "test")
    )

    logo_manifest.loc[is_unseen_ai, "split"] = "test"
    logo_manifest = logo_manifest[~is_other_generator_test_ai].reset_index(drop=True)

    return logo_manifest
