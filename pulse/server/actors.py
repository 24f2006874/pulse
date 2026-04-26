# server/actors.py
# Halluminate Bonus: Multi-Actor Environment
# 5 autonomous actors the agent must manage

import random


class NurseActor:
    """
    Actor 3: Reports vital signs each step.
    5 percent chance of misreporting one vital.
    Agent must learn to cross-check nurse reports.
    """

    ERROR_RATE = 0.05

    def report_vitals(self, true_vitals: dict) -> dict:
        reported = true_vitals.copy()
        had_error = False

        if random.random() < self.ERROR_RATE:
            vital = random.choice(list(reported.keys()))
            original = reported[vital]
            reported[vital] = round(original * random.uniform(0.85, 1.15), 1)
            had_error = True

        reported["nurse_note"] = (
            "Vitals reassessed."
            if not had_error
            else "Vitals rechecked - please verify."
        )

        return reported


class LabTechActor:
    """
    Actor 4: Processes diagnostic test orders.
    Some tests have processing delays.
    Agent must decide: wait for result or proceed?
    """

    PROCESSING_DELAYS = {
        "rapid_strep_test": 0,
        "urinalysis": 0,
        "blood_glucose": 0,
        "blood_gas": 0,
        "chest_xray": 0,
        "blood_test": 1,
        "urine_culture": 2,
        "blood_cultures": 2,
        "sputum_culture": 3,
        "urine_antigen": 1,
    }

    def __init__(self):
        self.pending = {}

    def order_test(
        self,
        test_name: str,
        current_step: int,
        result: str,
    ):
        delay = self.PROCESSING_DELAYS.get(test_name, 1)
        self.pending[test_name] = {
            "result": result,
            "ready_at": current_step + delay,
        }

    def get_available(self, current_step: int) -> dict:
        available = {}

        for test, data in list(self.pending.items()):
            if current_step >= data["ready_at"]:
                available[test] = data["result"]
                del self.pending[test]

        return available

    def get_pending_names(self) -> list:
        return list(self.pending.keys())


class AdminActor:
    """
    Actor 5: Hospital Administrator.
    Controls extra budget approval.
    Can approve or deny expensive tests.
    Replaces 'budget' as a passive resource.
    """

    APPROVAL_THRESHOLD = 300

    def __init__(self, initial_budget: float):
        self.initial_budget = initial_budget
        self.extra_granted = 0.0
        self.max_extra = initial_budget * 0.3

    def request_approval(
        self,
        test_cost: float,
        budget_remaining: float,
        clinical_justification: str,
    ) -> dict:
        """
        Agent can request extra budget from admin.
        Admin approves based on clinical justification quality.
        """
        if test_cost > budget_remaining:
            shortfall = test_cost - budget_remaining

            if shortfall > self.max_extra - self.extra_granted:
                return {
                    "approved": False,
                    "message": ("Request denied. Budget limit reached."),
                    "extra_granted": 0.0,
                }

            justification_ok = (
                shortfall <= self.APPROVAL_THRESHOLD or len(clinical_justification) > 20
            )

            if justification_ok:
                self.extra_granted += shortfall
                return {
                    "approved": True,
                    "message": (
                        f"Approved additional ${shortfall:.0f}. "
                        f"Clinical justification accepted."
                    ),
                    "extra_granted": shortfall,
                }
            else:
                return {
                    "approved": False,
                    "message": ("Denied. Insufficient clinical justification."),
                    "extra_granted": 0.0,
                }

        return {
            "approved": True,
            "message": "Within budget.",
            "extra_granted": 0.0,
        }
