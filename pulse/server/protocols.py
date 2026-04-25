# server/protocols.py
# The ground truth clinical protocols
# Hard-coded rules - no LLM judge needed

PROTOCOLS = {

    "uti": {
        "display_name": "Urinary Tract Infection",
        "difficulty": "easy",
        "budget": 500,
        "max_steps": 6,
        "deterioration_rate": 0.0,

        # Correct test ORDER matters
        "required_sequence": [
            "request_history",
            "urinalysis",
            "urine_culture",
        ],

        "available_tests": {
            "urinalysis": 50,
            "urine_culture": 80,
            "blood_test": 150,
            "ct_scan": 400,
            "ultrasound": 200,
        },

        "contraindications": [],

        # Multi-actor
        "specialist_specialty": "nephrology",
        "specialist_correct": "urinalysis",
        "specialist_wrong": "ct_scan",

        # Diagnosis
        "correct_diagnosis": "Urinary Tract Infection",
        "diagnosis_choices": [
            "Urinary Tract Infection",
            "Kidney Stone",
            "Pelvic Inflammatory Disease",
            "Interstitial Cystitis",
            "Vaginal Infection",
        ],

        "presentation": (
            "28-year-old female. Burning urination, "
            "increased frequency, lower abdominal pain for 2 days. "
            "No fever. Sexually active."
        ),

        "correct_department": "primary_care",
    },

    "strep_throat": {
        "display_name": "Streptococcal Pharyngitis",
        "difficulty": "easy",
        "budget": 400,
        "max_steps": 6,
        "deterioration_rate": 0.0,

        "required_sequence": [
            "request_history",
            "physical_exam",
            "rapid_strep_test",
        ],

        "available_tests": {
            "rapid_strep_test": 35,
            "throat_culture": 75,
            "mono_spot_test": 60,
            "blood_test": 150,
            "chest_xray": 200,
        },

        "contraindications": [
            "amoxicillin_without_strep_confirmed"
        ],

        "specialist_specialty": "ent",
        "specialist_correct": "rapid_strep_test",
        "specialist_wrong": "throat_culture",

        "correct_diagnosis": "Streptococcal Pharyngitis",
        "diagnosis_choices": [
            "Streptococcal Pharyngitis",
            "Viral Pharyngitis",
            "Mononucleosis",
            "Tonsillitis",
            "Epiglottitis",
        ],

        "presentation": (
            "16-year-old male. Severe sore throat, "
            "fever 38.8C, difficulty swallowing for 3 days. "
            "No cough. Tender neck lymph nodes."
        ),

        "correct_department": "primary_care",
    },

    "pneumonia": {
        "display_name": "Community-Acquired Pneumonia",
        "difficulty": "medium",
        "budget": 350,
        "max_steps": 10,
        "deterioration_rate": 2.0,

        "required_sequence": [
            "request_history",
            "physical_exam",
            "chest_xray",
            "blood_test",
        ],

        "available_tests": {
            "chest_xray": 150,
            "ct_chest": 800,
            "blood_test": 100,
            "sputum_culture": 80,
            "urine_antigen": 120,
        },

        "contraindications": [
            "ct_chest_before_xray"
        ],

        "specialist_specialty": "pulmonology",
        "specialist_correct": "chest_xray",
        "specialist_wrong": "ct_chest",

        "correct_diagnosis": "Community-Acquired Pneumonia",
        "diagnosis_choices": [
            "Community-Acquired Pneumonia",
            "Pulmonary Embolism",
            "Congestive Heart Failure",
            "COPD Exacerbation",
            "Lung Cancer",
        ],

        "presentation": (
            "67-year-old male with diabetes. "
            "Productive cough, fever 39.2C, "
            "shortness of breath, chest pain for 4 days. "
            "Oxygen saturation 92 percent on room air."
        ),

        "correct_department": "pulmonology",
    },

    "dka": {
        "display_name": "Diabetic Ketoacidosis",
        "difficulty": "medium",
        "budget": 350,
        "max_steps": 10,
        "deterioration_rate": 3.0,

        "required_sequence": [
            "request_history",
            "blood_glucose",
            "urine_ketones",
            "blood_gas",
            "electrolytes",
        ],

        "available_tests": {
            "blood_glucose": 20,
            "urine_ketones": 15,
            "blood_gas": 80,
            "electrolytes": 60,
            "hba1c": 50,
            "cardiac_enzymes": 200,
        },

        "contraindications": [
            "bicarbonate_if_ph_above_7"
        ],

        "specialist_specialty": "endocrinology",
        "specialist_correct": "blood_glucose",
        "specialist_wrong": "hba1c",

        "correct_diagnosis": "Diabetic Ketoacidosis",
        "diagnosis_choices": [
            "Diabetic Ketoacidosis",
            "Hyperosmolar Hyperglycemic State",
            "Alcoholic Ketoacidosis",
            "Lactic Acidosis",
            "Gastroenteritis",
        ],

        "presentation": (
            "22-year-old Type 1 diabetic. "
            "Nausea, vomiting, abdominal pain, "
            "fruity breath for 24 hours. "
            "Blood glucose finger-stick 480 mg/dL."
        ),

        "correct_department": "icu",
    },

    "sepsis": {
        "display_name": "Sepsis",
        "difficulty": "hard",
        "budget": 200,
        "max_steps": 15,
        "deterioration_rate": 5.0,

        "required_sequence": [
            "request_history",
            "blood_cultures",
            "lactate_level",
            "blood_test",
        ],

        "available_tests": {
            "blood_cultures": 80,
            "lactate_level": 40,
            "blood_test": 100,
            "urine_culture": 60,
            "chest_xray": 150,
            "ct_abdomen": 900,
        },

        "contraindications": [
            "antibiotics_before_blood_cultures"
        ],

        "specialist_specialty": "infectious_disease",
        "specialist_correct": "blood_cultures",
        "specialist_wrong": "ct_abdomen",

        "correct_diagnosis": "Sepsis",
        "diagnosis_choices": [
            "Sepsis",
            "Severe Urinary Tract Infection",
            "Pneumonia",
            "Meningitis",
            "Acute Abdomen",
        ],

        "presentation": (
            "72-year-old female from nursing home. "
            "Confusion, fever 39.8C, heart rate 118, "
            "blood pressure 88/60. "
            "2 days decreased oral intake. "
            "Suspected urinary source."
        ),

        "correct_department": "icu",
    },
}


def get_protocol(disease: str) -> dict:
    return PROTOCOLS.get(disease, {})


def get_diseases_by_difficulty(difficulty: str) -> list:
    return [
        d for d, p in PROTOCOLS.items()
        if p["difficulty"] == difficulty
    ]


def check_protocol_adherence(
    tests_ordered: list,
    required_sequence: list
) -> float:
    """
    LCS check: partial credit for correct order.
    Returns 0.0 to 1.0.
    """
    if not required_sequence:
        return 1.0
    if not tests_ordered:
        return 0.0

    protocol_idx = 0
    completed = 0

    for test in tests_ordered:
        if protocol_idx < len(required_sequence):
            if test == required_sequence[protocol_idx]:
                completed += 1
                protocol_idx += 1

    return round(completed / len(required_sequence), 2)


def check_contraindication(
    action: str,
    patient_flags: list,
    contraindications: list
) -> bool:
    """Returns True if action is contraindicated."""
    rules = {
        "amoxicillin_without_strep_confirmed": (
            "amoxicillin" in action and
            "strep_confirmed" not in patient_flags
        ),
        "ct_chest_before_xray": (
            action == "ct_chest" and
            "chest_xray" not in patient_flags
        ),
        "bicarbonate_if_ph_above_7": (
            action == "bicarbonate" and
            "low_ph" not in patient_flags
        ),
        "antibiotics_before_blood_cultures": (
            "antibiotic" in action and
            "blood_cultures" not in patient_flags
        ),
    }

    for rule in contraindications:
        if rule in rules and rules[rule]:
            return True
    return False