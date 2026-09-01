"""Stage 1 (variant annotation): labels variants with gene/functional context by calling
SnpEff via subprocess (standing in for ANNOVAR -- see config.py's module docstring for why).
Cohort-level: run once against the shared input VCF, not per patient.
"""

import subprocess
from pathlib import Path

from src.genomics_pipeline.config import SNPEFF_DATA_DIR, SNPEFF_GENOME_BUILD, resolve_snpeff
from src.genomics_pipeline.models import VariantAnnotation


def run_snpeff(vcf_path: Path, out_path: Path, genome_build: str = SNPEFF_GENOME_BUILD) -> Path:
    snpeff_bin = resolve_snpeff()
    data_dir = str(Path(SNPEFF_DATA_DIR).resolve())
    with open(out_path, "w") as out_f:
        result = subprocess.run(
            # -Xmx4g: the whole-genome GRCh37.75 database doesn't fit in SnpEff's 1GB default
            # JVM heap (loading it OOMs otherwise), even though the VCF itself is tiny.
            # -dataDir: see config.py's SNPEFF_DATA_DIR docstring.
            [snpeff_bin, "-Xmx4g", "-noStats", "-noLog", "-dataDir", data_dir, genome_build, str(vcf_path)],
            stdout=out_f,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"snpEff failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
    return out_path


def _parse_ann_field(info_field: str) -> tuple[str | None, str | None]:
    """Extracts (gene, consequence) from the top-ranked ANN= entry in a VCF INFO field.
    ANN format: Allele|Annotation|Annotation_Impact|Gene_Name|Gene_ID|... (SnpEff spec)."""
    for part in info_field.split(";"):
        if part.startswith("ANN="):
            entries = part[len("ANN="):].split(",")
            if not entries or not entries[0]:
                return None, None
            fields = entries[0].split("|")
            consequence = fields[1] if len(fields) > 1 and fields[1] else None
            gene = fields[3] if len(fields) > 3 and fields[3] else None
            return gene, consequence
    return None, None


def parse_annotated_vcf(annotated_vcf_path: Path, pipeline_version: str) -> list[VariantAnnotation]:
    annotations = []
    with open(annotated_vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos, variant_id, ref, alt, info = (
                fields[0], int(fields[1]), fields[2], fields[3], fields[4], fields[7]
            )
            if variant_id == ".":
                variant_id = f"{chrom}:{pos}:{ref}:{alt}"
            gene, consequence = _parse_ann_field(info)
            annotations.append(
                VariantAnnotation(
                    pipeline_version=pipeline_version,
                    chrom=chrom,
                    pos=pos,
                    variant_id=variant_id,
                    ref=ref,
                    alt=alt,
                    gene=gene,
                    consequence=consequence,
                )
            )
    return annotations


def annotate_variants(vcf_path: Path, out_dir: Path, pipeline_version: str) -> list[VariantAnnotation]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / "annotated.vcf"
    run_snpeff(vcf_path, annotated_path)
    return parse_annotated_vcf(annotated_path, pipeline_version)
