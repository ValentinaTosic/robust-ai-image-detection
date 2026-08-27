"""Leave-one-generator-out (LOGO) manifest construction.

Builds a manifest variant for testing generalization to an entirely unseen
AI generator: the held-out generator's AI images are removed from train and
validation -- so they cannot influence training, early stopping, or any
other model-selection decision -- and pooled into a dedicated test set
alongside the real test images.
"""

import pandas as pd


def list_ai_generators(manifest: pd.DataFrame) -> list[str]:
    """Return the sorted list of AI generator names present in a manifest.

    Args:
        manifest: Dataset manifest containing "label" and "generator" columns.

    Returns:
        Sorted generator names, excluding "real".
    """
    return sorted(manifest.loc[manifest["label"] == "ai", "generator"].unique())


def build_logo_manifest(manifest: pd.DataFrame, unseen_generator: str) -> pd.DataFrame:
    """Build a manifest variant that holds one AI generator fully unseen.

    The held-out generator's AI images -- wherever they originally sat
    (train, val, or test) -- are relabeled into the test split, pooling all
    of that generator's images (rather than just its original ~150 test
    images) for a more reliable test-set estimate. Other generators'
    original test-split AI images are dropped from the returned manifest,
    so the LOGO test set contains only real images plus the unseen
    generator -- not the usual seven-way test mix.

    Real images and every other generator's train/val AI images are left
    untouched, so training and model selection never see the unseen
    generator in any form.

    Args:
        manifest: Full dataset manifest (see ``AIImageDataset``), containing
            at least "label", "generator", and "split" columns.
        unseen_generator: Name of the AI generator to hold out, e.g.
            "midjourney". Must be a value present in the "generator" column
            for AI-labeled rows.

    Returns:
        A new manifest DataFrame (the input is not modified) with the
        unseen generator's AI rows relabeled to split="test" and other
        generators' original test-split AI rows removed.

    Raises:
        ValueError: If unseen_generator is not an AI generator present in
            the manifest.
    """
    available_generators = list_ai_generators(manifest)
    if unseen_generator not in available_generators:
        raise ValueError(
            f"Unknown unseen_generator {unseen_generator!r}; expected one of {available_generators}."
        )

    logo_manifest = manifest.copy()

    is_unseen_ai = (logo_manifest["label"] == "ai") & (logo_manifest["generator"] == unseen_generator)
    is_other_generator_test_ai = (
        (logo_manifest["label"] == "ai")
        & (logo_manifest["generator"] != unseen_generator)
        & (logo_manifest["split"] == "test")
    )

    logo_manifest.loc[is_unseen_ai, "split"] = "test"
    logo_manifest = logo_manifest[~is_other_generator_test_ai].reset_index(drop=True)

    return logo_manifest
