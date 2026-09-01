"""Stage 2 (GWAS) and Stage 3 (population stratification), both via PLINK.

This 1000 Genomes subset carries no real clinical phenotype -- there's no "pain" or any other
trait recorded for these samples in this context. A synthetic alternating case/control label
is assigned purely to give PLINK's --assoc something to test against and exercise the
association-testing machinery end-to-end. With 5 samples the test is statistically
underpowered by construction; treat p-values as illustrative of pipeline mechanics, not
evidence of real association -- the architecture doc itself flags small cohort size as a
known limitation for exactly this kind of analysis.
"""

import subprocess
from pathlib import Path

import pandas as pd

from src.genomics_pipeline.config import NUM_PRINCIPAL_COMPONENTS, resolve_plink
from src.genomics_pipeline.models import VariantAssociation


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result


def make_bed(vcf_path: Path, out_dir: Path, prefix: str = "cohort") -> Path:
    plink = resolve_plink()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [plink, "--vcf", str(vcf_path), "--make-bed", "--double-id", "--allow-extra-chr", "--out", prefix],
        cwd=out_dir,
    )
    return out_dir / prefix


def write_synthetic_phenotype(samples: list[str], out_dir: Path, prefix: str = "cohort") -> Path:
    """Alternating case(2)/control(1) labels in PLINK's phenotype-file format (FID IID PHENO).
    See module docstring: purely to exercise --assoc, not a real phenotype."""
    pheno_path = Path(out_dir) / f"{prefix}.pheno"
    with open(pheno_path, "w") as f:
        for i, sample in enumerate(samples):
            label = 2 if i % 2 == 0 else 1  # PLINK convention: 2=case, 1=control
            f.write(f"{sample} {sample} {label}\n")
    return pheno_path


def run_association_test(bed_prefix: Path, pheno_path: Path, out_dir: Path, prefix: str = "cohort_assoc") -> Path:
    plink = resolve_plink()
    out_dir = Path(out_dir)
    _run(
        [plink, "--bfile", str(bed_prefix), "--pheno", str(pheno_path), "--assoc", "--allow-no-sex", "--out", prefix],
        cwd=out_dir,
    )
    return out_dir / f"{prefix}.assoc"


def run_pca(bed_prefix: Path, out_dir: Path, prefix: str = "cohort_pca", num_pcs: int = NUM_PRINCIPAL_COMPONENTS) -> Path:
    plink = resolve_plink()
    out_dir = Path(out_dir)
    _run([plink, "--bfile", str(bed_prefix), "--pca", str(num_pcs), "--out", prefix], cwd=out_dir)
    return out_dir / f"{prefix}.eigenvec"


def parse_association_results(assoc_path: Path, pipeline_version: str) -> list[VariantAssociation]:
    df = pd.read_csv(assoc_path, sep=r"\s+")
    results = []
    for _, row in df.iterrows():
        p = row.get("P")
        results.append(
            VariantAssociation(
                pipeline_version=pipeline_version,
                chrom=str(row["CHR"]),
                variant_id=str(row["SNP"]),
                pos=int(row["BP"]),
                p_value=None if pd.isna(p) else float(p),
            )
        )
    return results


def parse_pca_results(eigenvec_path: Path) -> dict[str, list[float]]:
    """Returns {sample_id: [PC1, PC2, ...]}."""
    df = pd.read_csv(eigenvec_path, sep=r"\s+", header=None)
    num_pcs = df.shape[1] - 2
    df.columns = ["FID", "IID"] + [f"PC{i + 1}" for i in range(num_pcs)]
    return {row["IID"]: [float(row[f"PC{i + 1}"]) for i in range(num_pcs)] for _, row in df.iterrows()}
