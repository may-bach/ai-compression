"""
Clinical Critical-Error Diagnostic Probes.

A curated suite of prompt–expected-answer pairs designed to test whether
quantization corrupts clinically decisive tokens in three categories:

  1. **Dosage Fidelity**   — model must preserve exact numeric dosages.
  2. **Drug Identity**     — model must not confuse sound-alike medications.
  3. **Negation Retention** — model must preserve negation words that
                              invert the clinical meaning of a sentence.

These probes feed into the Clinical Critical-Error Rate (CCER) metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ClinicalProbe:
    """A single diagnostic probe."""
    category: str                  # "dosage" | "drug" | "negation"
    prompt: str                    # Input prompt for the model
    expected_tokens: List[str]     # Tokens that MUST appear in the output
    forbidden_tokens: List[str] = field(default_factory=list)  # Tokens that must NOT appear


# ════════════════════════════════════════════════════════════════════════
#  1.  Dosage Fidelity Probes
# ════════════════════════════════════════════════════════════════════════
DOSAGE_PROBES: List[ClinicalProbe] = [
    ClinicalProbe(
        category="dosage",
        prompt="What is the standard adult dose of acetaminophen for mild pain?",
        expected_tokens=["500", "1000", "mg"],
        forbidden_tokens=["5", "50", "5000"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the typical starting dose of metformin for type 2 diabetes?",
        expected_tokens=["500", "mg"],
        forbidden_tokens=["5", "50"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What dose of epinephrine is used in anaphylaxis for an adult?",
        expected_tokens=["0.3", "mg"],
        forbidden_tokens=["3", "30"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the standard prophylactic dose of enoxaparin?",
        expected_tokens=["40", "mg"],
        forbidden_tokens=["4", "400"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the maximum daily dose of ibuprofen for adults?",
        expected_tokens=["1200", "mg"],
        forbidden_tokens=["120", "12"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the loading dose of amiodarone for ventricular tachycardia?",
        expected_tokens=["150", "mg"],
        forbidden_tokens=["15", "1500"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the initial insulin dose for DKA management?",
        expected_tokens=["0.1", "units"],
        forbidden_tokens=["1", "10"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the pediatric dose of amoxicillin for acute otitis media?",
        expected_tokens=["mg", "kg"],
        forbidden_tokens=[],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the standard dose of ceftriaxone for bacterial meningitis?",
        expected_tokens=["2", "g"],
        forbidden_tokens=["20", "200"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the typical dose of furosemide IV for acute pulmonary edema?",
        expected_tokens=["40", "mg"],
        forbidden_tokens=["4", "400"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the maintenance dose of digoxin for atrial fibrillation?",
        expected_tokens=["0.125", "0.25", "mg"],
        forbidden_tokens=["1.25", "2.5"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What dose of naloxone is given IV for opioid overdose?",
        expected_tokens=["0.4", "mg"],
        forbidden_tokens=["4", "40"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the dose of aspirin for acute MI?",
        expected_tokens=["162", "325", "mg"],
        forbidden_tokens=["16", "32"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What is the standard adult dose of omeprazole?",
        expected_tokens=["20", "mg"],
        forbidden_tokens=["2", "200"],
    ),
    ClinicalProbe(
        category="dosage",
        prompt="What dose of vancomycin is used for MRSA bacteremia?",
        expected_tokens=["15", "mg", "kg"],
        forbidden_tokens=["1.5", "150"],
    ),
]

# ════════════════════════════════════════════════════════════════════════
#  2.  Drug Identity Probes
# ════════════════════════════════════════════════════════════════════════
DRUG_PROBES: List[ClinicalProbe] = [
    ClinicalProbe(
        category="drug",
        prompt="Which drug is first-line for hypertension: lisinopril or risperidone?",
        expected_tokens=["lisinopril"],
        forbidden_tokens=["risperidone"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Which inotrope is preferred for cardiogenic shock: dobutamine or dopamine at low dose?",
        expected_tokens=["dobutamine"],
        forbidden_tokens=[],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Name the reversal agent for heparin toxicity.",
        expected_tokens=["protamine"],
        forbidden_tokens=["naloxone", "flumazenil"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="What is the first-line antibiotic for community-acquired pneumonia in an outpatient?",
        expected_tokens=["azithromycin"],
        forbidden_tokens=["vancomycin", "meropenem"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Which medication is contraindicated in pregnancy: methotrexate or acetaminophen?",
        expected_tokens=["methotrexate"],
        forbidden_tokens=[],
    ),
    ClinicalProbe(
        category="drug",
        prompt="What drug is used for status epilepticus as first-line treatment?",
        expected_tokens=["lorazepam", "diazepam", "midazolam"],
        forbidden_tokens=["haloperidol"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Name the antidote for acetaminophen overdose.",
        expected_tokens=["acetylcysteine", "nac", "n-acetylcysteine"],
        forbidden_tokens=["naloxone"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Which statin has the highest potency for LDL reduction?",
        expected_tokens=["rosuvastatin"],
        forbidden_tokens=["pravastatin"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Which beta-blocker is cardioselective: metoprolol or propranolol?",
        expected_tokens=["metoprolol"],
        forbidden_tokens=[],
    ),
    ClinicalProbe(
        category="drug",
        prompt="What anticoagulant is preferred in heparin-induced thrombocytopenia (HIT)?",
        expected_tokens=["argatroban", "bivalirudin"],
        forbidden_tokens=["heparin", "enoxaparin"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Which antifungal is first-line for invasive candidiasis?",
        expected_tokens=["echinocandin", "caspofungin", "micafungin", "anidulafungin"],
        forbidden_tokens=["fluconazole"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="What is the vasopressor of choice for septic shock?",
        expected_tokens=["norepinephrine"],
        forbidden_tokens=["dopamine"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Name the reversal agent for warfarin in life-threatening bleeding.",
        expected_tokens=["vitamin k", "phytonadione", "kcentra", "pcc", "prothrombin"],
        forbidden_tokens=["protamine"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="Which SSRI is generally preferred as first-line for depression?",
        expected_tokens=["sertraline", "escitalopram"],
        forbidden_tokens=["haloperidol", "quetiapine"],
    ),
    ClinicalProbe(
        category="drug",
        prompt="What bronchodilator is used as a rescue inhaler in asthma?",
        expected_tokens=["albuterol", "salbutamol"],
        forbidden_tokens=["salmeterol", "tiotropium"],
    ),
]

# ════════════════════════════════════════════════════════════════════════
#  3.  Negation Retention Probes
# ════════════════════════════════════════════════════════════════════════
NEGATION_PROBES: List[ClinicalProbe] = [
    ClinicalProbe(
        category="negation",
        prompt='A CT scan report states: "There is no intracranial hemorrhage." Does the patient have intracranial hemorrhage?',
        expected_tokens=["no"],
        forbidden_tokens=["yes", "present", "confirmed"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='The clinical note says: "Patient denies chest pain." Is the patient experiencing chest pain?',
        expected_tokens=["no", "denies"],
        forbidden_tokens=["yes", "experiencing", "has"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='The lab report says: "Blood culture negative for growth." Is there a bacterial infection confirmed by culture?',
        expected_tokens=["no", "negative"],
        forbidden_tokens=["yes", "positive", "confirmed"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='The radiology report states: "Lungs are clear without consolidation." Is there pneumonia?',
        expected_tokens=["no", "without", "clear"],
        forbidden_tokens=["yes", "pneumonia confirmed"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='Progress note: "No evidence of metastatic disease on PET scan." Has cancer spread?',
        expected_tokens=["no"],
        forbidden_tokens=["yes", "metastatic", "spread"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='The pathology report reads: "Margins are free of tumor cells." Was the tumor completely removed?',
        expected_tokens=["yes", "free", "clear", "complete"],
        forbidden_tokens=["positive margins", "residual"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='ECG report: "No ST-segment elevation noted." Is the patient having a STEMI?',
        expected_tokens=["no"],
        forbidden_tokens=["yes", "STEMI confirmed"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='The note says: "Patient has not had any episodes of syncope." Has the patient fainted?',
        expected_tokens=["no", "not"],
        forbidden_tokens=["yes", "fainted"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='Discharge summary: "Wound shows no signs of infection." Is the wound infected?',
        expected_tokens=["no"],
        forbidden_tokens=["yes", "infected"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='Assessment: "Rules out pulmonary embolism based on CT angiography." Does the patient have PE?',
        expected_tokens=["no", "rules out", "ruled out"],
        forbidden_tokens=["yes", "confirmed"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='Lab: "HIV antibody test is non-reactive." Is the patient HIV positive?',
        expected_tokens=["no", "non-reactive", "negative"],
        forbidden_tokens=["yes", "positive", "reactive"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='Physical exam: "Abdomen is soft, non-tender, without guarding." Is there peritonitis?',
        expected_tokens=["no", "non-tender", "without"],
        forbidden_tokens=["yes", "peritonitis"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='The note reads: "Denies shortness of breath at rest or with exertion." Does the patient have dyspnea?',
        expected_tokens=["no", "denies"],
        forbidden_tokens=["yes", "dyspnea present"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='MRI report: "No ligamentous injury identified." Is there a ligament tear?',
        expected_tokens=["no"],
        forbidden_tokens=["yes", "tear", "rupture"],
    ),
    ClinicalProbe(
        category="negation",
        prompt='Allergy section: "No known drug allergies (NKDA)." Can this patient receive penicillin?',
        expected_tokens=["yes", "no known", "nkda"],
        forbidden_tokens=["allergic", "contraindicated"],
    ),
]


# ════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════

ALL_PROBES: List[ClinicalProbe] = DOSAGE_PROBES + DRUG_PROBES + NEGATION_PROBES


def get_probes_by_category(category: str) -> List[ClinicalProbe]:
    """Return probes of a given category ('dosage', 'drug', 'negation')."""
    return [p for p in ALL_PROBES if p.category == category]
