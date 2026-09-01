"""Locates the external genomics binaries (PLINK, SnpEff) this pipeline shells out to.

Neither is pip-installable. Both are resolved via PATH first (the normal case once Docker or a
dev machine has them installed properly), with PLINK_BIN/SNPEFF_BIN environment variable
overrides, and a fallback to the conda env this project's setup created them in locally
(see README.md's Genomics data section) so things work out of the box on a fresh checkout of
this machine without extra configuration.
"""

import os
import shutil
from pathlib import Path

PIPELINE_VERSION = "genomics-v0.1.0"

# ANNOVAR itself is gated behind manual registration on the author's site (no package-manager
# install path, non-commercial license) -- SnpEff stands in for it here: freely installable via
# bioconda, and one of the most widely used ANNOVAR-equivalent tools for the same job (variant
# -> gene/functional-consequence annotation).
SNPEFF_GENOME_BUILD = "GRCh37.75"  # 1000 Genomes VCFs are near-universally GRCh37-based

# SnpEff's own genome-database download (`snpEff download`) pulls from
# snpeff.blob.core.windows.net, which is unreachable in this project's build/dev environment (same
# public-mirror-unreachable situation as the VCF itself -- see PROGRESS.md). data/raw/genomics_snpeff_db/
# holds a local copy of the GRCh37.75 database, trimmed to chr21 only since this project's VCF never
# covers more than one chromosome (see vcf_loader.py); passed to SnpEff via -dataDir so it doesn't
# depend on SnpEff's own (potentially absent or differently-configured) default data directory.
# SNPEFF_DATA_DIR env var overrides this for a machine that already has a full local install.
SNPEFF_DATA_DIR = os.environ.get("SNPEFF_DATA_DIR", "data/raw/genomics_snpeff_db")

NUM_PRINCIPAL_COMPONENTS = 2  # only 5 samples -- PLINK caps meaningful PCs at n_samples - 1

PATHWAY_GMT_PATH = Path(__file__).resolve().parent / "pain_pathways.gmt"

_FALLBACK_BIN_DIRS = (
    "/opt/homebrew/Caskroom/miniforge/base/envs/genomics/bin",
    "/opt/conda/envs/genomics/bin",
    str(Path.home() / "miniconda3/envs/genomics/bin"),
)


def _resolve_binary(name: str, env_var: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for d in _FALLBACK_BIN_DIRS:
        candidate = Path(d) / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find '{name}' on PATH or in known conda env locations. "
        f"Set the {env_var} environment variable to its full path, or install it "
        f"(see README.md's Genomics data section)."
    )


def resolve_plink() -> str:
    return _resolve_binary("plink", "PLINK_BIN")


def resolve_snpeff() -> str:
    return _resolve_binary("snpEff", "SNPEFF_BIN")
