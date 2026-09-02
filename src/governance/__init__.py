# Deliberately does NOT import phi_check here: phi_check.py imports from src.emr_pipeline
# (reusing its PHI definitions), while src.emr_pipeline's own pipeline functions import
# log_audit_event below to log their own runs -- eagerly importing phi_check at this package's
# top level would make `src.governance` and `src.emr_pipeline` import each other's packages,
# a real circular-import hazard. Import `from src.governance.phi_check import ...` directly
# wherever it's needed instead (tests, verification scripts) -- unlike audit logging, PHI
# checking isn't something every other phase's pipeline needs to import.
from src.governance.audit import log_audit_event
from src.governance.fairness_check import SubgroupFairnessResult, check_subgroup_fairness
from src.governance.models import AuditLogEntry

__all__ = [
    "AuditLogEntry",
    "SubgroupFairnessResult",
    "check_subgroup_fairness",
    "log_audit_event",
]
