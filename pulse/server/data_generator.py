# server/data_generator.py
# Synthetic patient generator
# No real patient data. Pure Faker.

import random
from faker import Faker

fake = Faker()

VITALS_BY_DISEASE = {
    "uti": {
        "heart_rate": (70, 88),
        "blood_pressure": (110, 128),
        "oxygen_saturation": (96, 99),
        "temperature": (37.2, 38.3),
    },
    "strep_throat": {
        "heart_rate": (80, 100),
        "blood_pressure": (108, 128),
        "oxygen_saturation": (96, 99),
        "temperature": (38.3, 39.4),
    },
    "pneumonia": {
        "heart_rate": (88, 110),
        "blood_pressure": (98, 122),
        "oxygen_saturation": (88, 94),
        "temperature": (38.5, 39.8),
    },
    "dka": {
        "heart_rate": (100, 120),
        "blood_pressure": (92, 112),
        "oxygen_saturation": (94, 98),
        "temperature": (37.0, 38.5),
    },
    "sepsis": {
        "heart_rate": (112, 138),
        "blood_pressure": (78, 96),
        "oxygen_saturation": (88, 94),
        "temperature": (38.8, 40.2),
    },
}

TEST_RESULTS_BY_DISEASE = {
    "uti": {
        "urinalysis": "Positive: WBC 50/hpf, Bacteria 3+, Nitrites positive",
        "urine_culture": "E. coli 100,000 CFU/mL, sensitive to trimethoprim",
        "blood_test": "WBC 11,000 mild elevation",
        "ct_scan": "No renal calculi. Mild bladder wall thickening.",
    },
    "strep_throat": {
        "rapid_strep_test": "POSITIVE - Group A Streptococcus",
        "throat_culture": "Pending - results in 48 hours",
        "mono_spot_test": "Negative",
    },
    "pneumonia": {
        "chest_xray": "Right lower lobe consolidation - consistent with pneumonia",
        "blood_test": "WBC 18,000. CRP 120 elevated. Procalcitonin 2.1.",
        "ct_chest": "Multilobar pneumonia with small pleural effusion",
        "sputum_culture": "Streptococcus pneumoniae sensitive to amoxicillin",
    },
    "dka": {
        "blood_glucose": "Blood glucose 480 mg/dL severely elevated",
        "urine_ketones": "Large ketones 3+",
        "blood_gas": "pH 7.18. HCO3 8. pCO2 22. Metabolic acidosis.",
        "electrolytes": "Na 132. K 5.8 hyperkalemia. Monitor closely.",
        "hba1c": "HbA1c 11.2 percent - poorly controlled diabetes",
    },
    "sepsis": {
        "blood_cultures": "2 sets drawn. Results pending 24-48 hours.",
        "lactate_level": "Lactate 4.2 mmol/L - elevated. Tissue hypoperfusion.",
        "blood_test": "WBC 22,000. Bands 25 percent. Creatinine 2.8 - acute kidney injury.",
        "urine_culture": "Pending 48 hours.",
        "chest_xray": "Bilateral infiltrates - possible aspiration.",
        "ct_abdomen": "No acute abdominal pathology identified.",
    },
}


def generate_patient(disease: str) -> dict:
    vitals_range = VITALS_BY_DISEASE.get(disease, VITALS_BY_DISEASE["uti"])

    vitals = {
        k: round(random.uniform(*v), 1)
        for k, v in vitals_range.items()
    }

    return {
        "patient_id": f"PT-{fake.uuid4()[:8].upper()}",
        "age": random.randint(18, 80),
        "sex": random.choice(["male", "female"]),
        "disease": disease,
        "vitals": vitals,
        "flags": [],
    }


def get_test_result(disease: str, test_name: str) -> str:
    results = TEST_RESULTS_BY_DISEASE.get(disease, {})
    return results.get(test_name, f"{test_name}: Result pending")