from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from pulse.models import PulseAction, PulseObservation, PulseState

class PulseEnv(EnvClient[PulseAction, PulseObservation, PulseState]):
    def _step_payload(self, action: PulseAction) -> Dict:
        return {
            "action_type": action.action_type,
            "content": action.content,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PulseObservation]:
        obs_data = payload.get("observation", {})
        observation = PulseObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
            patient_summary=obs_data.get("patient_summary", ""),
            available_tests=obs_data.get("available_tests", []),
            test_results=obs_data.get("test_results", {}),
            diagnosis_options=obs_data.get("diagnosis_options", []),
            specialist_recommendation=obs_data.get("specialist_recommendation"),
            specialist_confidence=obs_data.get("specialist_confidence", 0.0),
            specialist_reasoning=obs_data.get("specialist_reasoning", ""),
            specialist_updated=obs_data.get("specialist_updated", False),
            specialist_update_reason=obs_data.get("specialist_update_reason", ""),
            nurse_vitals_report=obs_data.get("nurse_vitals_report", {}),
            lab_pending_results=obs_data.get("lab_pending_results", []),
            admin_message=obs_data.get("admin_message", ""),
            schema_alert=obs_data.get("schema_alert", ""),
            schema_change_type=obs_data.get("schema_change_type", ""),
            steps_remaining=obs_data.get("steps_remaining", 15),
            budget_remaining=obs_data.get("budget_remaining", 500.0),
            heart_rate=obs_data.get("heart_rate", 80.0),
            blood_pressure=obs_data.get("blood_pressure", 120.0),
            oxygen_saturation=obs_data.get("oxygen_saturation", 98.0),
            temperature=obs_data.get("temperature", 37.0),
            clinical_notes=obs_data.get("clinical_notes", ""),
            warning=obs_data.get("warning", ""),
            reward_protocol=obs_data.get("reward_protocol", 0.0),
            reward_diagnosis=obs_data.get("reward_diagnosis", 0.0),
            reward_specialist=obs_data.get("reward_specialist", 0.0),
            reward_efficiency=obs_data.get("reward_efficiency", 0.0),
            reward_safety=obs_data.get("reward_safety", 0.0),
            reward_anti_exploit=obs_data.get("reward_anti_exploit", 0.0),
            reward_token_efficiency=obs_data.get("reward_token_efficiency", 0.0),
            reward_schema_adaptation=obs_data.get("reward_schema_adaptation", 0.0),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> PulseState:
        return PulseState(**payload)