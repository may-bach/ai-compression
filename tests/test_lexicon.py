"""
Unit tests for the Clinical Lexicon — token classification and weighting.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clinical_lexicon import (
    classify_token,
    get_token_weight,
    is_clinical_token,
    weight_token_set,
)


class TestDrugDetection:
    def test_common_drugs(self):
        for drug in ["metformin", "amoxicillin", "lisinopril", "morphine", "warfarin"]:
            assert is_clinical_token(drug), f"Failed to detect drug: {drug}"
            assert classify_token(drug) == "drug"

    def test_case_insensitive(self):
        assert is_clinical_token("Metformin")
        assert is_clinical_token("AMOXICILLIN")

    def test_non_drug(self):
        assert not is_clinical_token("the")
        assert not is_clinical_token("patient")
        assert classify_token("however") == "generic"


class TestDosageDetection:
    def test_standard_dosages(self):
        assert is_clinical_token("500mg")
        assert is_clinical_token("0.25 mcg")
        assert is_clinical_token("10 mL")
        assert is_clinical_token("1000 IU")

    def test_dosage_classification(self):
        assert classify_token("500mg") == "dosage"
        assert classify_token("0.5 ml") == "dosage"


class TestNegationDetection:
    def test_negation_cues(self):
        for neg in ["no", "not", "denies", "without", "negative", "absent"]:
            assert is_clinical_token(neg), f"Failed to detect negation: {neg}"
            assert classify_token(neg) == "negation"

    def test_negation_phrases(self):
        assert is_clinical_token("no evidence")
        assert is_clinical_token("rules out")


class TestLabValues:
    def test_lab_keywords(self):
        for lab in ["creatinine", "hemoglobin", "troponin", "potassium", "glucose"]:
            assert is_clinical_token(lab), f"Failed to detect lab: {lab}"
            assert classify_token(lab) == "lab_value"

    def test_lab_units(self):
        assert is_clinical_token("mmol/l")
        assert is_clinical_token("mg/dl")


class TestWeighting:
    def test_clinical_weight(self):
        assert get_token_weight("morphine", alpha=5.0) == 5.0
        assert get_token_weight("500mg", alpha=5.0) == 5.0
        assert get_token_weight("no", alpha=5.0) == 5.0

    def test_generic_weight(self):
        assert get_token_weight("the", alpha=5.0) == 1.0
        assert get_token_weight("patient", alpha=5.0) == 1.0

    def test_custom_alpha(self):
        assert get_token_weight("morphine", alpha=10.0) == 10.0

    def test_weight_token_set(self):
        tokens = ["morphine", "the", "500mg", "is"]
        weights = weight_token_set(tokens, alpha=5.0)
        assert weights["morphine"] == 5.0
        assert weights["the"] == 1.0
        assert weights["500mg"] == 5.0
        assert weights["is"] == 1.0
