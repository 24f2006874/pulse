# server/rewards.py
# 6 Core Rewards + 2 Bonus Rewards
# All programmatic. Zero LLM judges.
# Mercor Bonus: Token efficiency
# Patronus Bonus: Schema adaptation

from typing import List, Dict


def reward_protocol_sequence(
    tests_ordered: List[str],
    required_sequence: List[str],
) -> float:
    """
    R1: Protocol Adherence (28% weight)
    LCS check. Partial credit for progress.
    """
    from .protocols import check_protocol_adherence
    return check_protocol_adherence(tests_ordered, required_sequence)


def reward_diagnosis_correct(
    submitted: str,
    correct: str,
) -> float:
    """
    R2: Diagnosis Correctness (22% weight)
    Multiple choice exact match. Binary.
    Zero string parsing. Zero LLM.
    """
    if not submitted:
        return 0.0
    return 1.0 if submitted == correct else 0.0


def reward_specialist_handling(
    consulted: bool,
    specialist_was_correct: bool,
    agent_overrode: bool,
) -> float:
    """
    R3: Specialist Handling (15% weight)
    The betrayal moment reward.

    Trust correct = 1.0
    Override wrong = 1.0
    Override correct = 0.0
    Follow wrong = 0.0
    Not consulted = 0.3
    """
    if not consulted:
        return 0.3

    if specialist_was_correct and not agent_overrode:
        return 1.0

    if not specialist_was_correct and agent_overrode:
        return 1.0

    return 0.0


def reward_efficiency(
    budget_used: float,
    initial_budget: float,
    steps_used: int,
    max_steps: int,
) -> float:
    """
    R4: Resource Efficiency (12% weight)
    Budget + time efficiency combined.
    """
    if initial_budget <= 0 or max_steps <= 0:
        return 0.5

    budget_score = max(0.0, 1.0 - (budget_used / initial_budget))
    step_score = max(0.0, 1.0 - (steps_used / max_steps))

    return round((budget_score + step_score) / 2, 2)


def reward_patient_safety(
    contraindication_triggered: bool,
    patient_critical: bool,
) -> float:
    """
    R5: Patient Safety (8% weight)
    Binary: safe or unsafe.
    """
    if contraindication_triggered:
        return 0.0
    if patient_critical:
        return 0.2
    return 1.0


def reward_anti_exploit(
    tests_ordered: List[str],
    actions_taken: List[str],
    steps_used: int,
    submitted_diagnosis: str,
) -> float:
    """
    R6: Anti-Exploit Detection (5% weight)
    If triggered: zeroes total reward.
    """
    if len(tests_ordered) >= 3:
        if len(set(tests_ordered[-3:])) == 1:
            return 0.0

    if submitted_diagnosis and steps_used <= 1:
        return 0.0

    if len(tests_ordered) > 10:
        return 0.0

    return 1.0


def reward_token_efficiency(
    reasoning: str,
    action_correct: bool,
    max_tokens: int = 50,
) -> float:
    """
    R7: Token Efficiency - Mercor Bonus (5% weight)
    Correct + concise = 1.0
    Correct + verbose = reduced
    Wrong = 0.0
    """
    if not action_correct:
        return 0.0

    if not reasoning:
        return 0.8

    token_count = len(reasoning.split())

    if token_count <= max_tokens:
        return 1.0

    penalty = (token_count - max_tokens) / max_tokens
    return max(0.3, round(1.0 - penalty, 2))


def reward_schema_adaptation(
    schema_changed: bool,
    agent_adapted: bool,
) -> float:
    """
    R8: Schema Adaptation - Patronus AI Bonus (5% weight)
    Adapts to protocol drift mid-episode.
    """
    if not schema_changed:
        return 1.0

    return 1.0 if agent_adapted else 0.0


def compute_total_reward(
    tests_ordered: List[str],
    required_sequence: List[str],
    submitted_diagnosis: str,
    correct_diagnosis: str,
    specialist_consulted: bool,
    specialist_was_correct: bool,
    agent_overrode: bool,
    budget_used: float,
    initial_budget: float,
    steps_used: int,
    max_steps: int,
    contraindication_triggered: bool,
    patient_critical: bool,
    actions_taken: List[str],
    reasoning: str = "",
    schema_changed: bool = False,
    agent_adapted: bool = False,
) -> Dict[str, float]:
    """
    Combine all 8 rewards.
    All logged separately for W&B monitoring.
    """
    r1 = reward_protocol_sequence(tests_ordered, required_sequence)
    r2 = reward_diagnosis_correct(submitted_diagnosis, correct_diagnosis)
    r3 = reward_specialist_handling(
        specialist_consulted,
        specialist_was_correct,
        agent_overrode
    )
    r4 = reward_efficiency(budget_used, initial_budget, steps_used, max_steps)
    r5 = reward_patient_safety(contraindication_triggered, patient_critical)
    r6 = reward_anti_exploit(tests_ordered, actions_taken, steps_used, submitted_diagnosis)
    # Token efficiency: requires action to be part of successful protocol (r1) 
    # or correct diagnosis (r2). This is a proxy for "action usefulness".
    # A more precise metric would track per-action ground truth.
    action_useful = (r1 >= 0.5 or r2 > 0.0)
    r7 = reward_token_efficiency(reasoning, action_useful)
    r8 = reward_schema_adaptation(schema_changed, agent_adapted)

    weighted = (
        r1 * 0.28 +
        r2 * 0.22 +
        r3 * 0.15 +
        r4 * 0.12 +
        r5 * 0.08 +
        r6 * 0.05 +
        r7 * 0.05 +
        r8 * 0.05
    )

    total = weighted * r6
    total = round(max(0.0, min(1.0, total)), 3)

    return {
        "total": total,
        "protocol": r1,
        "diagnosis": r2,
        "specialist": r3,
        "efficiency": r4,
        "safety": r5,
        "anti_exploit": r6,
        "token_efficiency": r7,
        "schema_adaptation": r8,
    }