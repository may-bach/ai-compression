"""
Clinical token lexicon for CW-LSS weight function w(t).

Tokens in four clinical categories receive weight alpha (default 5.0):
  drug names, dosage numerals/units, lab values, negation cues.
All other tokens receive weight 1.0.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

DRUG_NAMES: Set[str] = {
    "acetaminophen", "paracetamol", "ibuprofen", "aspirin", "naproxen",
    "morphine", "fentanyl", "tramadol", "codeine", "oxycodone",
    "hydrocodone", "ketorolac", "diclofenac", "celecoxib",
    "amoxicillin", "azithromycin", "ciprofloxacin", "metronidazole",
    "doxycycline", "ceftriaxone", "vancomycin", "clindamycin",
    "levofloxacin", "penicillin", "erythromycin", "trimethoprim",
    "sulfamethoxazole", "meropenem", "piperacillin", "tazobactam",
    "gentamicin", "nitrofurantoin",
    "atenolol", "metoprolol", "lisinopril", "enalapril", "amlodipine",
    "losartan", "valsartan", "hydrochlorothiazide", "furosemide",
    "spironolactone", "digoxin", "warfarin", "heparin", "enoxaparin",
    "clopidogrel", "atorvastatin", "simvastatin", "rosuvastatin",
    "nitroglycerin", "diltiazem", "verapamil", "amiodarone",
    "metformin", "insulin", "glipizide", "glyburide", "empagliflozin",
    "sitagliptin", "semaglutide", "liraglutide", "levothyroxine",
    "prednisone", "prednisolone", "dexamethasone", "hydrocortisone",
    "sertraline", "fluoxetine", "escitalopram", "venlafaxine",
    "duloxetine", "bupropion", "quetiapine", "risperidone",
    "olanzapine", "haloperidol", "lorazepam", "diazepam",
    "alprazolam", "clonazepam", "gabapentin", "pregabalin",
    "levetiracetam", "phenytoin", "carbamazepine", "valproic",
    "lithium", "lamotrigine",
    "albuterol", "salbutamol", "ipratropium", "fluticasone",
    "budesonide", "montelukast", "theophylline",
    "omeprazole", "pantoprazole", "lansoprazole", "ranitidine",
    "ondansetron", "metoclopramide", "lactulose", "loperamide",
    "acyclovir", "oseltamivir", "fluconazole", "itraconazole",
    "tacrolimus", "cyclosporine", "mycophenolate", "azathioprine",
    "rituximab", "infliximab", "adalimumab",
    "allopurinol", "colchicine", "sildenafil", "epinephrine",
    "atropine", "naloxone", "dopamine", "dobutamine", "norepinephrine",
    "vasopressin", "phenylephrine",
}

DOSAGE_UNITS: List[str] = [
    "mg", "mcg", "µg", "g", "kg", "ml", "mL", "L", "IU",
    "units", "mmol", "mEq", "cc", "drops", "tabs", "capsules",
    "tablets", "puffs", "sprays", "patches",
]

_units_pattern = "|".join(re.escape(u) for u in DOSAGE_UNITS)
DOSAGE_RE = re.compile(
    rf"(?<!\w)(\d[\d,]*(?:\.\d+)?)[\s\-]*({_units_pattern})(?:/(?:{_units_pattern}))?(?!\w)",
    re.IGNORECASE,
)

NUMERIC_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?!\w)")

LAB_KEYWORDS: Set[str] = {
    "mmol/l", "meq/l", "mg/dl", "ng/dl", "ng/ml", "pg/ml",
    "µg/dl", "mcg/dl", "g/dl", "g/l", "u/l", "iu/l",
    "mmhg", "bpm", "breaths/min", "cells/µl", "cells/ul",
    "x10^9/l", "x10^3/ul", "x10^6/ul", "%",
    "hemoglobin", "hematocrit", "creatinine", "bilirubin",
    "troponin", "albumin", "glucose", "potassium", "sodium",
    "calcium", "magnesium", "phosphate", "chloride", "bicarbonate",
    "bun", "gfr", "egfr", "hba1c", "a1c", "tsh", "t3", "t4",
    "alt", "ast", "alp", "ggt", "ldh", "bnp", "crp", "esr",
    "inr", "ptt", "aptt", "fibrinogen", "d-dimer",
    "wbc", "rbc", "platelets", "neutrophils", "lymphocytes",
    "monocytes", "eosinophils", "basophils", "reticulocytes",
    "ferritin", "transferrin", "tibc", "iron",
    "ph", "pao2", "paco2", "spo2", "fio2",
    "lactate", "ammonia", "lipase", "amylase",
}

NEGATION_CUES: Set[str] = {
    "no", "not", "none", "nor", "never", "neither",
    "without", "absence", "absent", "negative",
    "denies", "denied", "deny", "denying",
    "unremarkable", "normal",
    "rules out", "ruled out", "rule out",
    "no evidence", "no sign", "no signs",
    "unlikely", "improbable",
    "free of", "free from",
    "resolved", "cleared",
}


def is_clinical_token(token: str) -> bool:
    """Return True if token is clinically salient."""
    t = token.strip().lower()
    return (
        t in DRUG_NAMES
        or t in NEGATION_CUES
        or t in LAB_KEYWORDS
        or bool(DOSAGE_RE.search(token))
    )


def get_token_weight(token: str, alpha: float = 5.0) -> float:
    """Return alpha for clinically salient tokens, 1.0 otherwise."""
    return alpha if is_clinical_token(token) else 1.0


def classify_token(token: str) -> str:
    """Return the clinical category of a token, or 'generic'."""
    t = token.strip().lower()
    if t in DRUG_NAMES:
        return "drug"
    if t in NEGATION_CUES:
        return "negation"
    if t in LAB_KEYWORDS:
        return "lab_value"
    if DOSAGE_RE.search(token):
        return "dosage"
    return "generic"


def weight_token_set(tokens: List[str], alpha: float = 5.0) -> Dict[str, float]:
    """Return {token: w(t)} for a list of tokens."""
    return {t: get_token_weight(t, alpha) for t in tokens}
