# server/pulse_environment.py
# PULSE Main Environment
# All themes + all bonus prizes integrated

import copy
import uuid
import random
from typing import Optional

from openenv.core.env_server.interfaces import Environment

try:
    from pulse.models import PulseAction, PulseObservation, PulseState
except ModuleNotFoundError:
    from models import PulseAction, PulseObservation, PulseState
from .protocols import (
    PROTOCOLS,
    get_diseases_by_difficulty,
    check_contraindication,
)
from .specialist import SpecialistAgent
from .actors import NurseActor, LabTechActor, AdminActor
from .rewards import compute_total_reward
from .curriculum import PulseCurriculum
from .data_generator import generate_patient, get_test_result


# Constants for schema drift (Patronus AI Bonus)
SCHEMA_DRIFT_EVENTS = [
    {
        "trigger_step": 4,
        "diseases": ["pneumonia"],
        "change_type": "new_contraindication",
        "message": (
            "Hospital Policy Update: Azithromycin now requires "
            "cardiac clearance before prescribing."
        ),
    },
    {
        "trigger_step": 5,
        "diseases": ["sepsis"],
        "change_type": "protocol_update",
        "message": (
            "Sepsis Bundle Update: Lactate measurement now "
            "required before antibiotic administration."
        ),
    },
    {
        "trigger_step": 6,
        "diseases": ["dka"],
        "change_type": "formulary_change",
        "message": (
            "Pharmacy Update: Regular insulin now requires "
            "attending physician approval."
        ),
    },
]

# Schema drift configuration
SCHEMA_DRIFT_STEPS = [4, 5, 6]
SCHEMA_DRIFT_CHANCE = 0.10

# Patient deterioration thresholds
CRITICAL_O2_THRESHOLD = 85
CRITICAL_BP_THRESHOLD = 70
CRITICAL_HR_THRESHOLD = 145
MAX_HEART_RATE = 160
MIN_BLOOD_PRESSURE = 60
MIN_OXYGEN_SATURATION = 70


class PulseEnvironment(Environment):
    """
    PULSE: Protocol Understanding for Life-Saving Evidence

    Multi-agent clinical protocol environment.

    Themes covered:
    - Theme 1: Multi-Agent (specialist + nurse + lab + admin)
    - Theme 2: Long-Horizon (12-15 step protocols)
    - Theme 3: World Modeling (patient deterioration)
    - Theme 4: Self-Improvement (adaptive curriculum)

    Bonus prizes:
    - Snorkel AI: Specialist updates mid-episode with reason
    - Halluminate: 5 autonomous actors
    - Mercor: Token efficiency reward
    - Patronus AI: Schema drift mid-episode
    - Scale AI: Healthcare enterprise workflow framing
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Valid action types for validation
    VALID_ACTION_TYPES = {
        "request_history",
        "physical_exam",
        "order_test",
        "administer_treatment",
        "consult_specialist",
        "override_specialist",
        "request_admin_approval",
        "acknowledge_schema_change",
        "submit_diagnosis",
        "wait",
    }

    def __init__(self):
        super().__init__()
        self.curriculum = PulseCurriculum()
        self._state = None
        self._protocol = None
        self._patient = None
        self._specialist = None
        self._nurse = None
        self._lab_tech = None
        self._admin = None
        self._specialist_result = None
        self._schema_changed_this_episode = False
        self._agent_adapted_to_schema = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        disease: Optional[str] = None,
        **kwargs,
    ) -> PulseObservation:
        """Start new clinical episode."""

        if seed is not None:
            random.seed(seed)

        diff = (difficulty or self.curriculum.get_stage()["name"]).strip().lower()

        if disease:
            chosen_disease = disease
        elif difficulty:
            # When UI explicitly sets difficulty, sample from that bucket
            # instead of the curriculum's current stage.
            difficulty_pool = get_diseases_by_difficulty(diff)
            chosen_disease = random.choice(difficulty_pool) if difficulty_pool else self.curriculum.sample_disease()
        else:
            chosen_disease = self.curriculum.sample_disease()

        protocol = copy.deepcopy(PROTOCOLS.get(chosen_disease, {}))

        self._patient = generate_patient(chosen_disease)
        self._protocol = protocol

        self._specialist = SpecialistAgent(
            specialty=protocol.get("specialist_specialty", "infectious_disease"),
            difficulty=diff,
        )
        self._nurse = NurseActor()
        self._lab_tech = LabTechActor()
        self._admin = AdminActor(protocol.get("budget", 500))

        self._schema_changed_this_episode = False
        self._agent_adapted_to_schema = False

        self._state = PulseState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            correct_disease=chosen_disease,
            correct_protocol=protocol.get("required_sequence", []),
            tests_ordered=[],
            actions_taken=[],
            specialist_consulted=False,
            specialist_was_wrong=False,
            agent_overrode=False,
            last_specialist_rec="",
            nurse_error_this_episode=False,
            admin_approved_extra=False,
            drift_events=[],
            agent_adapted_to_drift=False,
            current_vitals=self._patient["vitals"].copy(),
            patient_flags=[],
            budget_used=0.0,
            patient_critical=False,
            difficulty=diff,
            last_action="",
            consecutive_same=0,
        )

        nurse_report = self._nurse.report_vitals(
            self._state.current_vitals
        )

        # Track nurse errors for agent learning
        if "Vitals rechecked - please verify" in nurse_report.get("nurse_note", ""):
            self._state.nurse_error_this_episode = True

        return PulseObservation(
            done=False,
            reward=0.0,
            patient_summary=self._format_summary(),
            available_tests=list(protocol.get("available_tests", {}).keys()),
            test_results={},
            diagnosis_options=protocol.get("diagnosis_choices", []),
            specialist_recommendation=None,
            specialist_confidence=0.0,
            specialist_reasoning="",
            specialist_updated=False,
            specialist_update_reason="",
            nurse_vitals_report=nurse_report,
            lab_pending_results=[],
            admin_message="",
            schema_alert="",
            schema_change_type="",
            steps_remaining=protocol.get("max_steps", 10),
            budget_remaining=self._budget_remaining(),
            heart_rate=self._state.current_vitals.get("heart_rate", 80),
            blood_pressure=self._state.current_vitals.get("blood_pressure", 120),
            oxygen_saturation=self._state.current_vitals.get("oxygen_saturation", 98),
            temperature=self._state.current_vitals.get("temperature", 37),
            clinical_notes=(
                f"New patient. {protocol.get('presentation', '')}"
            ),
            warning="",
        )

    def step(
        self,
        action: PulseAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> PulseObservation:
        """Process one clinical action."""
        if self._state is None or self._protocol is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self._state.step_count += 1
        self._state.actions_taken.append(action.action_type)

        if action.action_type == self._state.last_action:
            self._state.consecutive_same += 1
        else:
            self._state.consecutive_same = 0
        self._state.last_action = action.action_type

        response = {
            "clinical_notes": "",
            "warning": "",
            "test_results": {},
            "specialist_recommendation": None,
            "specialist_confidence": 0.0,
            "specialist_reasoning": "",
            "specialist_updated": False,
            "specialist_update_reason": "",
            "admin_message": "",
            "schema_alert": "",
            "schema_change_type": "",
        }

        submitted_diagnosis = ""
        done = False

        self._deteriorate_patient()

        schema_alert = self._check_schema_drift()
        if schema_alert:
            response["schema_alert"] = schema_alert["message"]
            response["schema_change_type"] = schema_alert["change_type"]
            self._schema_changed_this_episode = True

        if self._state.patient_critical:
            done = True
            response["warning"] = "PATIENT CRITICAL - Episode ending"

        max_steps = self._protocol.get("max_steps", 10)
        if self._state.step_count >= max_steps:
            done = True
            response["clinical_notes"] += " [MAX STEPS REACHED]"

        if not done:
            done, response, submitted_diagnosis = self._process_action(
                action, response
            )

        specialist_status = self._specialist.get_status()
        response["specialist_status"] = specialist_status

        contraindication_triggered = "CONTRAINDICATION_VIOLATED" in self._state.patient_flags

        rewards = compute_total_reward(
            tests_ordered=self._state.tests_ordered,
            required_sequence=self._state.correct_protocol,
            submitted_diagnosis=submitted_diagnosis,
            correct_diagnosis=self._protocol.get("correct_diagnosis", ""),
            specialist_consulted=self._state.specialist_consulted,
            specialist_was_correct=not self._state.specialist_was_wrong,
            agent_overrode=self._state.agent_overrode,
            budget_used=self._state.budget_used,
            initial_budget=self._protocol.get("budget", 500),
            steps_used=self._state.step_count,
            max_steps=max_steps,
            contraindication_triggered=contraindication_triggered,
            patient_critical=self._state.patient_critical,
            actions_taken=self._state.actions_taken,
            reasoning=action.reasoning,
            schema_changed=self._schema_changed_this_episode,
            agent_adapted=self._agent_adapted_to_schema,
        )

        if done:
            self.curriculum.record(rewards["total"])

        nurse_report = self._nurse.report_vitals(
            self._state.current_vitals
        )

        available_results = self._lab_tech.get_available(
            self._state.step_count
        )

        return PulseObservation(
            done=done,
            reward=rewards["total"],
            patient_summary=self._format_summary(),
            available_tests=list(
                self._protocol.get("available_tests", {}).keys()
            ),
            test_results={
                **response.get("test_results", {}),
                **available_results,
            },
            diagnosis_options=self._protocol.get("diagnosis_choices", []),
            specialist_recommendation=response.get("specialist_recommendation"),
            specialist_confidence=response.get("specialist_confidence", 0.0),
            specialist_reasoning=response.get("specialist_reasoning", ""),
            specialist_updated=response.get("specialist_updated", False),
            specialist_update_reason=response.get("specialist_update_reason", ""),
            nurse_vitals_report=nurse_report,
            lab_pending_results=self._lab_tech.get_pending_names(),
            admin_message=response.get("admin_message", ""),
            schema_alert=response.get("schema_alert", ""),
            schema_change_type=response.get("schema_change_type", ""),
            steps_remaining=max(0, max_steps - self._state.step_count),
            budget_remaining=self._budget_remaining(),
            heart_rate=self._state.current_vitals.get("heart_rate", 80),
            blood_pressure=self._state.current_vitals.get("blood_pressure", 120),
            oxygen_saturation=self._state.current_vitals.get("oxygen_saturation", 98),
            temperature=self._state.current_vitals.get("temperature", 37),
            clinical_notes=response.get("clinical_notes", ""),
            warning=response.get("warning", ""),
            reward_protocol=rewards["protocol"],
            reward_diagnosis=rewards["diagnosis"],
            reward_specialist=rewards["specialist"],
            reward_efficiency=rewards["efficiency"],
            reward_safety=rewards["safety"],
            reward_anti_exploit=rewards["anti_exploit"],
            reward_token_efficiency=rewards["token_efficiency"],
            reward_schema_adaptation=rewards["schema_adaptation"],
        )

    @property
    def state(self) -> PulseState:
        return self._state

    def _process_action(
        self,
        action: PulseAction,
        response: dict,
    ):
        done = False
        submitted = ""

        action_type = action.action_type
        content = action.content

        # Enforce schema drift as real action constraint
        # If schema has changed but not acknowledged, restrict to acknowledgment only
        if self._is_schema_drift_constraint_active() and action_type != "acknowledge_schema_change":
            response["warning"] = (
                "Schema drift detected! You must acknowledge the protocol update "
                "before taking other actions. Use 'acknowledge_schema_change'."
            )
            response["clinical_notes"] = (
                "Protocol update pending. Action blocked until acknowledged."
            )
            return done, response, submitted

        # Validate action type
        if action_type not in self.VALID_ACTION_TYPES:
            response["warning"] = f"Invalid action type: {action_type}"
            response["clinical_notes"] = (
                f"Unknown action '{action_type}'. "
                f"Valid actions: {sorted(self.VALID_ACTION_TYPES)}"
            )
            return done, response, submitted

        if action_type == "request_history":
            response["clinical_notes"] = (
                f"History: {self._protocol.get('presentation', '')}"
            )
            self._state.tests_ordered.append("request_history")

        elif action_type == "physical_exam":
            vitals = self._state.current_vitals
            response["clinical_notes"] = (
                f"Exam: HR {vitals.get('heart_rate', 80):.0f} "
                f"BP {vitals.get('blood_pressure', 120):.0f} "
                f"O2 {vitals.get('oxygen_saturation', 98):.0f}% "
                f"Temp {vitals.get('temperature', 37):.1f}C"
            )
            self._state.tests_ordered.append("physical_exam")

        elif action_type == "order_test":
            test = content
            costs = self._protocol.get("available_tests", {})

            if test in costs:
                cost = costs[test]
                
                # Check budget before ordering
                budget_remaining = self._budget_remaining()
                if cost > budget_remaining:
                    response["warning"] = (
                        f"Insufficient budget for {test}. "
                        f"Cost: ${cost}, Remaining: ${budget_remaining:.0f}. "
                        f"Request admin approval for additional funds."
                    )
                    response["clinical_notes"] = (
                        f"Cannot order {test} - budget exceeded. "
                        f"Consider requesting admin approval."
                    )
                else:
                    contraindicated = check_contraindication(
                        test,
                        self._state.patient_flags,
                        self._protocol.get("contraindications", [])
                    )
                    
                    if contraindicated:
                        self._state.patient_flags.append("CONTRAINDICATION_VIOLATED")
                        response["warning"] = f"CONTRAINDICATION ALERT: {test} not allowed in current patient state"
                        response["clinical_notes"] = f"Cannot order {test} due to contraindication"
                    else:
                        self._state.budget_used += cost
                        self._state.tests_ordered.append(test)
                        self._state.patient_flags.append(test)

                        result = get_test_result(
                            self._state.correct_disease, test
                        )

                        self._lab_tech.order_test(
                            test, self._state.step_count, result
                        )

                        response["clinical_notes"] = (
                            f"Ordered: {test}. Cost ${cost}. "
                            f"Result: {result}"
                        )

                    if self._state.specialist_consulted:
                        update = self._specialist.update_recommendation(
                            new_evidence={"latest_test": test},
                            correct_rec=self._protocol.get("specialist_correct", ""),
                            wrong_rec=self._protocol.get("specialist_wrong", ""),
                        )
                        if update["updated"]:
                            response["specialist_updated"] = True
                            response["specialist_update_reason"] = update["reason"]
                            response["specialist_recommendation"] = update["recommendation"]
                            if not update.get("is_correct", True):
                                self._state.specialist_was_wrong = True

            else:
                response["clinical_notes"] = f"Test {test} not available."
                response["warning"] = (
                    f"Available: {list(costs.keys())}"
                )

        elif action_type == "consult_specialist":
            self._state.specialist_consulted = True

            result = self._specialist.recommend(
                full_state={
                    "vitals": self._state.current_vitals,
                    "chief_complaint": self._protocol.get("presentation", ""),
                    "symptoms": [],
                },
                correct_rec=self._protocol.get("specialist_correct", ""),
                wrong_rec=self._protocol.get("specialist_wrong", ""),
            )

            if not result["is_correct"]:
                self._state.specialist_was_wrong = True
                self._state.last_specialist_rec = result["recommendation"]

            response["specialist_recommendation"] = result["recommendation"]
            response["specialist_confidence"] = result["confidence"]
            response["specialist_reasoning"] = result["reasoning"]
            response["clinical_notes"] = (
                f"Specialist ({self._specialist.specialty}): "
                f"{result['recommendation']} "
                f"(confidence {result['confidence']:.0%})"
            )

        elif action_type == "override_specialist":
            self._state.agent_overrode = True
            self._agent_adapted_to_schema = True
            response["clinical_notes"] = (
                f"Overriding specialist. Reason: {action.reasoning}"
            )

        elif action_type == "request_admin_approval":
            try:
                test_cost = float(content.strip())
            except (TypeError, ValueError, AttributeError):
                test_cost = 0.0

            budget_remaining = self._budget_remaining()

            approval = self._admin.request_approval(
                test_cost=test_cost,
                budget_remaining=budget_remaining,
                clinical_justification=action.reasoning,
            )

            if approval["approved"] and approval.get("extra_granted", 0.0) > 0:
                self._state.admin_approved_extra = True

            response["admin_message"] = approval["message"]
            response["clinical_notes"] = f"Admin: {approval['message']}"

        elif action_type == "administer_treatment":
            treatment = content.strip()
            if not treatment:
                response["warning"] = "No treatment specified"
                response["clinical_notes"] = "Treatment action requires medication/treatment name."
            else:
                contraindicated = check_contraindication(
                    treatment,
                    self._state.patient_flags,
                    self._protocol.get("contraindications", []),
                )

                if contraindicated:
                    self._state.patient_flags.append("CONTRAINDICATION_VIOLATED")
                    response["warning"] = (
                        f"CONTRAINDICATION ALERT: {treatment} is unsafe in current patient state"
                    )
                    response["clinical_notes"] = f"Treatment blocked: {treatment}"
                else:
                    self._state.patient_flags.append(treatment)
                    response["clinical_notes"] = f"Treatment administered: {treatment}"

        elif action_type == "acknowledge_schema_change":
            self._agent_adapted_to_schema = True
            response["clinical_notes"] = (
                "Protocol update acknowledged. Adjusting approach."
            )

        elif action_type == "submit_diagnosis":
            submitted = content
            done = True
            
            # Validate diagnosis is from available options
            diagnosis_options = self._protocol.get("diagnosis_choices", [])
            if diagnosis_options and submitted not in diagnosis_options:
                response["warning"] = (
                    f"Diagnosis '{submitted}' not in available options: {diagnosis_options}"
                )
                response["clinical_notes"] = (
                    f"Invalid diagnosis: '{submitted}'. "
                    f"Choose from: {diagnosis_options}"
                )
                submitted = ""  # Clear submission so it doesn't count as correct
            else:
                correct = submitted == self._protocol.get("correct_diagnosis", "")
                response["clinical_notes"] = (
                    f"Diagnosis: {submitted}. "
                    f"{'CORRECT' if correct else 'INCORRECT'}. "
                    f"Answer: {self._protocol.get('correct_diagnosis', '')}"
                )

        elif action_type == "wait":
            response["clinical_notes"] = "Waiting. Patient deteriorates."
            response["warning"] = "Avoid waiting - costs time and health."

        return done, response, submitted

    def _deteriorate_patient(self):
        rate = self._protocol.get("deterioration_rate", 0.0)
        if rate <= 0:
            return

        vitals = self._state.current_vitals

        vitals["heart_rate"] = min(
            MAX_HEART_RATE, vitals.get("heart_rate", 80) + rate * 0.4
        )
        vitals["blood_pressure"] = max(
            MIN_BLOOD_PRESSURE, vitals.get("blood_pressure", 120) - rate
        )
        vitals["oxygen_saturation"] = max(
            MIN_OXYGEN_SATURATION, vitals.get("oxygen_saturation", 98) - rate * 0.5
        )

        o2 = vitals.get("oxygen_saturation", 98)
        bp = vitals.get("blood_pressure", 120)
        hr = vitals.get("heart_rate", 80)

        if o2 < CRITICAL_O2_THRESHOLD or bp < CRITICAL_BP_THRESHOLD or hr > CRITICAL_HR_THRESHOLD:
            self._state.patient_critical = True

        self._state.current_vitals = vitals

    def _check_schema_drift(self) -> Optional[dict]:
        """
        Patronus AI Bonus: Schema drift check.
        10 percent chance at steps 4, 5, 6.
        """
        step = self._state.step_count
        disease = self._state.correct_disease

        if step not in SCHEMA_DRIFT_STEPS:
            return None

        if random.random() > SCHEMA_DRIFT_CHANCE:
            return None

        for event in SCHEMA_DRIFT_EVENTS:
            if (disease in event.get("diseases", []) and
                    event["trigger_step"] == step):
                self._state.drift_events.append(event["change_type"])
                return {
                    "change_type": event["change_type"],
                    "message": event["message"],
                }

        return None

    def _is_schema_drift_constraint_active(self) -> bool:
        """Check if schema drift has occurred but not been acknowledged."""
        return self._schema_changed_this_episode and not self._agent_adapted_to_schema

    def _format_summary(self) -> str:
        if not self._protocol:
            return "No active patient"

        vitals = self._state.current_vitals
        stage = (
            getattr(self._state, "difficulty", "")
            or self._protocol.get("difficulty", "")
            or self.curriculum.get_stage()["name"]
        ).upper()
        budget_remaining = self._budget_remaining()

        return (
            f"[{stage}] {self._protocol.get('display_name', '')} | "
            f"HR: {vitals.get('heart_rate', 80):.0f} | "
            f"BP: {vitals.get('blood_pressure', 120):.0f} | "
            f"O2: {vitals.get('oxygen_saturation', 98):.0f}% | "
            f"Budget: ${budget_remaining:.0f}"
        )

    def _budget_remaining(self) -> float:
        base_budget = self._protocol.get("budget", 500) if self._protocol else 500
        admin_extra = self._admin.extra_granted if self._admin else 0.0
        return max(0.0, (base_budget + admin_extra) - self._state.budget_used)
