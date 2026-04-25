# server/curriculum.py
# Adaptive difficulty scaling
# Self-Improvement Theme (Theme 4)

import random


class PulseCurriculum:
    """
    Threshold-based adaptive curriculum.
    Escalates difficulty as agent improves.
    De-escalates if agent struggles.
    """

    STAGES = [
        {
            "name": "easy",
            "diseases": ["uti", "strep_throat"],
            "specialist_conflict_rate": 0.0,
            "deterioration_rate": 0.0,
            "escalate_threshold": 0.75,
            "window": 10,
        },
        {
            "name": "medium",
            "diseases": ["pneumonia", "dka"],
            "specialist_conflict_rate": 0.2,
            "deterioration_rate": 2.0,
            "escalate_threshold": 0.70,
            "window": 10,
        },
        {
            "name": "hard",
            "diseases": ["sepsis"],
            "specialist_conflict_rate": 0.3,
            "deterioration_rate": 5.0,
            "escalate_threshold": None,
            "window": 10,
        },
    ]

    def __init__(self):
        self.current_stage_idx = 0
        self.episode_rewards = []

    def get_stage(self) -> dict:
        return self.STAGES[self.current_stage_idx]

    def sample_disease(self) -> str:
        stage = self.get_stage()
        return random.choice(stage["diseases"])

    def record(self, reward: float) -> bool:
        self.episode_rewards.append(reward)
        return self._check_escalation()

    def _check_escalation(self) -> bool:
        stage = self.get_stage()
        window = stage["window"]
        threshold = stage["escalate_threshold"]

        if len(self.episode_rewards) < window:
            return False

        recent = self.episode_rewards[-window:]
        avg = sum(recent) / len(recent)

        if (threshold is not None and
                avg >= threshold and
                self.current_stage_idx < len(self.STAGES) - 1):
            self.current_stage_idx += 1
            print(
                f"PULSE CURRICULUM: Escalating to "
                f"{self.STAGES[self.current_stage_idx]['name']} "
                f"(avg reward: {avg:.2f})"
            )
            return True

        if avg < 0.30 and self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            print(
                f"PULSE CURRICULUM: De-escalating to "
                f"{self.STAGES[self.current_stage_idx]['name']} "
                f"(avg reward: {avg:.2f})"
            )

        return False

    def stats(self) -> dict:
        recent = self.episode_rewards[-10:] if self.episode_rewards else []
        return {
            "stage": self.get_stage()["name"],
            "total_episodes": len(self.episode_rewards),
            "recent_avg": round(sum(recent) / len(recent), 3) if recent else 0.0,
        }