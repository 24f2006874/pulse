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
    Not consulted = 0.1 (reduced to prevent 'never consult' exploit)
    """
    if not consulted:
        return 0.1

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
    Hardened to detect:
    - Repeated action loops (same action 3+ times in window)
    - Repeated tests beyond tail-only (any 3+ repeats in sequence)
    - Early diagnosis gaming (diagnosis before meaningful work)
    - Excessive testing without progression
    """
    # Check for repeated action loops (same action 3+ times in sliding window)
    if len(actions_taken) >= 3:
        for i in range(len(actions_taken) - 2):
            if actions_taken[i] == actions_taken[i+1] == actions_taken[i+2]:
                return 0.0

    # Check for repeated tests beyond tail-only (any 3+ same tests in sequence)
    if len(tests_ordered) >= 3:
        for i in range(len(tests_ordered) - 2):
            if tests_ordered[i] == tests_ordered[i+1] == tests_ordered[i+2]:
                return 0.0

    # Early diagnosis gaming: diagnosis before meaningful protocol progress
    # Require at least 2 distinct actions OR 1 test + 1 non-test action before diagnosis
    if submitted_diagnosis:
        # Count meaningful actions (tests, history, physical exam, etc.)
        meaningful_actions = [a for a in actions_taken if a in [
            "request_history", "physical_exam", "order_test"
        ]]
        if len(meaningful_actions) < 2:
            return 0.0

    # Excessive testing without progression (more than 8 tests total)
    if len(tests_ordered) > 8:
        return 0.0

    # Test repetition penalty: more than 2 repeats of any single test
    if len(tests_ordered) >= 3:
        test_counts = {}
        for test in tests_ordered:
            test_counts[test] = test_counts.get(test, 0) + 1
            if test_counts[test] > 2:
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