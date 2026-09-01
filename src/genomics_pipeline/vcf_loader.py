"""Reads the input VCF: sample IDs (used as patient_id, same convention as the FHIR Patient.id
/ wearable subject-folder identifiers in the other two pipelines) and a per-variant, per-sample
genotype dosage table used later to build genuinely per-patient pathway scores.

Expected input layout: data/raw/genomics/<anything>.vcf or .vcf.gz -- a standard VCF, one file,
subset to the 5 sample columns and (for a tractable demo) a single chromosome. No specific
filename required; the loader globs for the extension.
"""

import gzip
from pathlib import Path

import pandas as pd

VARIANT_ID_COLUMNS = ("CHROM", "POS", "ID", "REF", "ALT")


def discover_vcf(raw_dir) -> Path:
    raw_dir = Path(raw_dir)
    candidates = sorted(list(raw_dir.glob("*.vcf")) + list(raw_dir.glob("*.vcf.gz")))
    if not candidates:
        raise FileNotFoundError(
            f"No .vcf or .vcf.gz file found in {raw_dir}. Expected a 1000 Genomes VCF subset "
            "to 5 samples on one chromosome."
        )
    return candidates[0].resolve()


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def load_samples(vcf_path: Path) -> list[str]:
    with _open(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
    raise ValueError(f"{vcf_path} has no #CHROM header line")


def _gt_dosage(gt_field: str) -> float:
    gt = gt_field.split(":")[0]
    alleles = gt.replace("|", "/").split("/")
    if any(a == "." for a in alleles):
        return float("nan")
    return float(sum(1 for a in alleles if a != "0"))


def load_variant_table(vcf_path: Path) -> pd.DataFrame:
    """One row per variant: CHROM, POS, ID, REF, ALT, plus one 0/1/2 dosage column per sample
    (NaN where genotype is missing)."""
    samples = load_samples(vcf_path)
    rows = []
    with _open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos, variant_id, ref, alt = fields[0], int(fields[1]), fields[2], fields[3], fields[4]
            if variant_id == ".":
                variant_id = f"{chrom}:{pos}:{ref}:{alt}"
            dosages = [_gt_dosage(gt) for gt in fields[9:]]
            row = {"CHROM": chrom, "POS": pos, "ID": variant_id, "REF": ref, "ALT": alt}
            row.update(dict(zip(samples, dosages)))
            rows.append(row)
    return pd.DataFrame(rows)
