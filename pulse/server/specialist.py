# server/specialist.py
# Snorkel AI Bonus: Expert-in-the-Loop
# Specialist with partial observability
# Updates recommendations when new evidence arrives

import random


SPECIALTY_VITALS = {
    "cardiology": ["heart_rate", "blood_pressure"],
    "pulmonology": ["oxygen_saturation", "temperature"],
    "nephrology": ["blood_pressure"],
    "endocrinology": ["temperature"],
    "infectious_disease": ["temperature", "heart_rate"],
    "ent": ["temperature"],
}

COMPETENCE_BY_DIFFICULTY = {
    "easy": 1.0,
    "medium": 0.75,
    "hard": 0.70,
}


class SpecialistAgent:
    """
    Snorkel AI Bonus: Simulated expert-in-the-loop.

    Key behaviors:
    - Only sees specialty-relevant vitals (partial observability)
    - Competence degrades with difficulty
    - Updates recommendation when new evidence arrives
    - Provides reason for any update (appears in observation)
    """

    def __init__(self, specialty: str, difficulty: str):
        self.specialty = specialty
        self.difficulty = difficulty
        self.competence = COMPETENCE_BY_DIFFICULTY.get(difficulty, 0.7)
        self.was_consulted = False
        self.last_recommendation = None
        self.last_was_correct = None

    def observe(self, full_state: dict) -> dict:
        """
        Partial observability:
        Specialist only sees their specialty vitals.
        """
        relevant = SPECIALTY_VITALS.get(self.specialty, [])
        all_vitals = full_state.get("vitals", {})

        return {
            "vitals": {
                k: v for k, v in all_vitals.items()
                if k in relevant
            },
            "chief_complaint": full_state.get("chief_complaint", ""),
            "symptoms": full_state.get("symptoms", [])[:2],
        }

    def recommend(
        self,
        full_state: dict,
        correct_rec: str,
        wrong_rec: str,
    ) -> dict:
        """
        Generate initial recommendation.
        Correct with probability = competence.
        """
        self.was_consulted = True

        is_correct = random.random() < self.competence

        if is_correct:
            rec = correct_rec
            confidence = round(random.uniform(0.80, 0.95), 2)
            reasoning = (
                f"Based on {self.specialty} assessment: "
                f"{rec} is the appropriate next step."
            )
        else:
            rec = wrong_rec
            confidence = round(random.uniform(0.65, 0.80), 2)
            reasoning = (
                f"Based on {self.specialty} assessment: "
                f"{rec} may provide additional clarity."
            )

        self.last_recommendation = rec
        self.last_was_correct = is_correct

        return {
            "recommendation": rec,
            "confidence": confidence,
            "reasoning": reasoning,
            "is_correct": is_correct,
        }

    def update_recommendation(
        self,
        new_evidence: dict,
        correct_rec: str,
        wrong_rec: str,
    ) -> dict:
        """
        Snorkel AI Bonus Core:
        Specialist updates when new evidence arrives.
        Reason MUST appear in observation for agent to learn.
        """
        new_test = new_evidence.get("latest_test", "")

        if not new_test or not self.was_consulted:
            return {
                "updated": False,
                "recommendation": self.last_recommendation,
                "reason": "",
            }

        # New evidence may change specialist opinion
        changed = random.random() < 0.4

        if changed:
            is_correct = random.random() < self.competence

            if is_correct:
                new_rec = correct_rec
            else:
                new_rec = wrong_rec

            reason = (
                f"Updated recommendation based on {new_test} results. "
                f"Now recommending {new_rec} instead of "
                f"{self.last_recommendation}."
            )

            self.last_recommendation = new_rec
            self.last_was_correct = is_correct

            return {
                "updated": True,
                "recommendation": new_rec,
                "reason": reason,
                "is_correct": is_correct,
            }

        return {
            "updated": False,
            "recommendation": self.last_recommendation,
            "reason": "",
        }

    def get_status(self) -> dict:
        return {
            "was_consulted": self.was_consulted,
            "was_correct": self.last_was_correct,
            "last_recommendation": self.last_recommendation,
        }