from typing import List, Optional, Dict, Union, Any
from pydantic import Field
from openenv.core.env_server.interfaces import (
    Action,
    Observation,
    State
)


class PulseAction(Action):
    """
    One action the AI takes per step.

    Examples:
        PulseAction(action_type="request_history")
        PulseAction(action_type="order_test", content="chest_xray")
        PulseAction(action_type="submit_diagnosis", content="Pneumonia")
    """
    action_type: str = "wait"
    content: str = ""
    reasoning: str = ""


class PulseObservation(Observation):
    """
    What the AI sees after each action.
    Inherits done and reward from base class.
    """

    # Patient information
    patient_summary: str = ""
    available_tests: List[str] = Field(default_factory=list)
    test_results: Dict[str, str] = Field(default_factory=dict)
    diagnosis_options: List[str] = Field(default_factory=list)

    # Specialist (Snorkel AI bonus)
    specialist_recommendation: Optional[str] = None
    specialist_confidence: float = 0.0
    specialist_reasoning: str = ""
    specialist_updated: bool = False
    specialist_update_reason: str = ""

    # Multi-actor reports (Halluminate bonus)
    nurse_vitals_report: Dict[str, Union[float, str]] = Field(default_factory=dict)
    lab_pending_results: List[str] = Field(default_factory=list)
    admin_message: str = ""

    # Schema drift alert (Patronus AI bonus)
    schema_alert: str = ""
    schema_change_type: str = ""

    # Progress
    steps_remaining: int = 15
    budget_remaining: float = 500.0

    # Vitals (deteriorate over time)
    heart_rate: float = 80.0
    blood_pressure: float = 120.0
    oxygen_saturation: float = 98.0
    temperature: float = 37.0

    # Feedback
    clinical_notes: str = ""
    warning: str = ""

    # Reward breakdown (all 8 components for W&B)
    reward_protocol: float = 0.0
    reward_diagnosis: float = 0.0
    reward_specialist: float = 0.0
    reward_efficiency: float = 0.0
    reward_safety: float = 0.0
    reward_anti_exploit: float = 0.0
    reward_token_efficiency: float = 0.0
    reward_schema_adaptation: float = 0.0


class PulseState(State):
    """
    Internal environment memory.
    AI cannot access this directly.
    """
    episode_id: str = ""
    step_count: int = 0

    # Hidden ground truth
    correct_disease: str = ""
    correct_protocol: List[str] = Field(default_factory=list)

    # Episode tracking
    tests_ordered: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)

    # Specialist tracking
    specialist_consulted: bool = False
    specialist_was_wrong: bool = False
    agent_overrode: bool = False
    last_specialist_rec: str = ""

    # Actor tracking
    nurse_error_this_episode: bool = False
    admin_approved_extra: bool = False

    # Schema drift tracking (Patronus AI)
    drift_events: List[str] = Field(default_factory=list)
    agent_adapted_to_drift: bool = False

    # Patient state
    current_vitals: Dict[str, float] = Field(default_factory=dict)
    patient_flags: List[str] = Field(default_factory=list)
    budget_used: float = 0.0
    patient_critical: bool = False

    # Curriculum
    difficulty: str = "easy"

    # Anti-exploit
    last_action: str = ""
    consecutive_same: int = 0
