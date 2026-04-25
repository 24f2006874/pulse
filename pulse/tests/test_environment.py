# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive tests for PULSE Environment.
Run with: python -m pytest pulse/tests/test_environment.py -v
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pulse.models import PulseAction, PulseObservation, PulseState
from pulse.server.pulse_environment import PulseEnvironment
from pulse.server.protocols import PROTOCOLS, check_protocol_adherence, check_contraindication
from pulse.server.specialist import SpecialistAgent
from pulse.server.actors import NurseActor, LabTechActor, AdminActor
from pulse.server.rewards import compute_total_reward
from pulse.server.curriculum import PulseCurriculum
from pulse.server.data_generator import generate_patient, get_test_result


class TestPulseEnvironment:
    """Test suite for PulseEnvironment."""

    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test."""
        return PulseEnvironment()

    @pytest.fixture
    def easy_env(self, env):
        """Environment reset with easy difficulty."""
        env.reset(difficulty="easy", disease="uti")
        return env

    def test_environment_creation(self, env):
        """Test that environment can be created."""
        assert env is not None
        assert env.SUPPORTS_CONCURRENT_SESSIONS is True

    def test_reset_returns_observation(self, env):
        """Test that reset returns a valid observation."""
        obs = env.reset(difficulty="easy", disease="uti")
        assert isinstance(obs, PulseObservation)
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.patient_summary != ""
        assert len(obs.available_tests) > 0
        assert len(obs.diagnosis_options) > 0

    def test_reset_with_seed(self, env):
        """Test that seeding produces consistent results."""
        obs1 = env.reset(seed=42, disease="uti")
        obs2 = env.reset(seed=42, disease="uti")
        assert obs1.patient_summary == obs2.patient_summary
        assert obs1.heart_rate == obs2.heart_rate

    def test_step_before_reset_raises_clear_error(self, env):
        """Test that step before reset fails with a clear message."""
        with pytest.raises(RuntimeError, match="Environment must be reset before calling step"):
            env.step(PulseAction(action_type="wait"))

    def test_observation_inherits_done_and_reward(self):
        """Test done/reward remain usable via inherited Observation fields."""
        obs = PulseObservation(
            done=False,
            reward=0.0,
            patient_summary="Test patient",
        )
        assert obs.done is False
        assert obs.reward == 0.0

    def test_valid_action_types(self, env):
        """Test that all valid action types are accepted."""
        env.reset(difficulty="easy", disease="uti")
        
        valid_actions = [
            PulseAction(action_type="request_history"),
            PulseAction(action_type="physical_exam"),
            PulseAction(action_type="wait"),
        ]
        
        for action in valid_actions:
            obs = env.step(action)
            assert obs is not None
            # Should not have invalid action warning
            assert "Invalid action type" not in obs.warning

    def test_invalid_action_type(self, env):
        """Test that invalid action types are rejected."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="invalid_action")
        obs = env.step(action)
        
        assert "Invalid action type" in obs.warning
        assert "Unknown action" in obs.clinical_notes

    def test_request_history(self, env):
        """Test request_history action."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="request_history")
        obs = env.step(action)
        
        assert "History:" in obs.clinical_notes
        assert "request_history" in env.state.tests_ordered

    def test_physical_exam(self, env):
        """Test physical_exam action."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="physical_exam")
        obs = env.step(action)
        
        assert "Exam:" in obs.clinical_notes
        assert "HR" in obs.clinical_notes
        assert "physical_exam" in env.state.tests_ordered

    def test_order_test_success(self, env):
        """Test ordering a valid test."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="order_test", content="urinalysis")
        obs = env.step(action)
        
        assert "Ordered:" in obs.clinical_notes
        assert "urinalysis" in env.state.tests_ordered
        assert env.state.budget_used == 50  # Cost of urinalysis

    def test_order_test_not_available(self, env):
        """Test ordering an unavailable test."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="order_test", content="mri_scan")
        obs = env.step(action)
        
        assert "not available" in obs.clinical_notes
        assert "Available:" in obs.warning

    def test_order_test_budget_exceeded(self, env):
        """Test ordering a test that exceeds budget."""
        # Create env with very small budget
        env.reset(difficulty="easy", disease="uti")
        env._protocol["budget"] = 10  # Very small budget
        
        action = PulseAction(action_type="order_test", content="ct_scan")  # Costs 400
        obs = env.step(action)
        
        assert "Insufficient budget" in obs.warning or "budget exceeded" in obs.clinical_notes

    def test_consult_specialist(self, env):
        """Test consulting the specialist."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="consult_specialist")
        obs = env.step(action)
        
        assert obs.specialist_recommendation is not None
        assert obs.specialist_confidence > 0
        assert obs.specialist_reasoning != ""
        assert env.state.specialist_consulted is True

    def test_override_specialist(self, env):
        """Test overriding the specialist."""
        env.reset(difficulty="easy", disease="uti")
        env._state.specialist_consulted = True  # Must consult first
        
        action = PulseAction(action_type="override_specialist", reasoning="Evidence contradicts")
        obs = env.step(action)
        
        assert env.state.agent_overrode is True
        assert "Overriding specialist" in obs.clinical_notes

    def test_submit_diagnosis_valid(self, env):
        """Test submitting a valid diagnosis."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="submit_diagnosis", content="Urinary Tract Infection")
        obs = env.step(action)
        
        assert obs.done is True
        assert "CORRECT" in obs.clinical_notes

    def test_submit_diagnosis_invalid_option(self, env):
        """Test submitting a diagnosis not in options."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="submit_diagnosis", content="Random Disease")
        obs = env.step(action)
        
        assert obs.done is True
        assert "Invalid diagnosis" in obs.warning or "not in available options" in obs.warning

    def test_submit_diagnosis_incorrect(self, env):
        """Test submitting an incorrect but valid diagnosis."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="submit_diagnosis", content="Kidney Stone")
        obs = env.step(action)
        
        assert obs.done is True
        assert "INCORRECT" in obs.clinical_notes

    def test_wait_action(self, env):
        """Test wait action."""
        env.reset(difficulty="easy", disease="uti")
        
        action = PulseAction(action_type="wait")
        obs = env.step(action)
        
        assert "Waiting" in obs.clinical_notes
        assert "deteriorates" in obs.clinical_notes

    def test_max_steps_ends_episode(self, env):
        """Test that max steps ends the episode."""
        env.reset(difficulty="easy", disease="uti")
        # UTI has max_steps=6
        
        for i in range(6):
            action = PulseAction(action_type="wait")
            obs = env.step(action)
        
        assert obs.done is True
        assert "MAX STEPS REACHED" in obs.clinical_notes

    def test_patient_deterioration(self, env):
        """Test that patient vitals deteriorate over time."""
        env.reset(difficulty="medium", disease="pneumonia")
        
        initial_o2 = env.state.current_vitals["oxygen_saturation"]
        
        # Take several steps
        for _ in range(5):
            action = PulseAction(action_type="wait")
            env.step(action)
        
        final_o2 = env.state.current_vitals["oxygen_saturation"]
        assert final_o2 < initial_o2

    def test_budget_tracking(self):
        """Test that budget is tracked correctly."""
        env = PulseEnvironment()
        env.reset(difficulty="easy", disease="uti")
        
        initial_budget = env._protocol.get("budget", 500)
        
        # Order a test
        action = PulseAction(action_type="order_test", content="urinalysis")
        obs = env.step(action)
        
        assert obs.budget_remaining == initial_budget - 50
        
        obs = env.step(PulseAction(action_type="wait"))
        assert obs.budget_remaining == initial_budget - 50

    def test_state_property(self, env):
        """Test that state property returns PulseState."""
        env.reset(difficulty="easy", disease="uti")
        state = env.state
        assert isinstance(state, PulseState)
        assert state.episode_id != ""
        assert state.step_count == 0

    def test_contraindication_check(self, env):
        """Test contraindication detection."""
        # Test strep throat contraindication
        env.reset(difficulty="easy", disease="strep_throat")
        
        # First, try to order amoxicillin without confirming strep
        # (This would be caught by check_contraindication if "amoxicillin" is in the action)
        # For now, just verify the protocol has contraindications
        assert len(env._protocol.get("contraindications", [])) > 0


class TestSpecialistAgent:
    """Test suite for SpecialistAgent."""

    def test_specialist_creation(self):
        """Test specialist can be created."""
        specialist = SpecialistAgent(specialty="cardiology", difficulty="easy")
        assert specialist.specialty == "cardiology"
        assert specialist.competence == 1.0  # Easy = 100% competent

    def test_specialist_competence_by_difficulty(self):
        """Test competence varies by difficulty."""
        easy_spec = SpecialistAgent(specialty="cardiology", difficulty="easy")
        med_spec = SpecialistAgent(specialty="cardiology", difficulty="medium")
        hard_spec = SpecialistAgent(specialty="cardiology", difficulty="hard")
        
        assert easy_spec.competence == 1.0
        assert med_spec.competence == 0.75
        assert hard_spec.competence == 0.70

    def test_specialist_recommend(self):
        """Test specialist generates recommendations."""
        specialist = SpecialistAgent(specialty="cardiology", difficulty="easy")
        
        result = specialist.recommend(
            full_state={"vitals": {"heart_rate": 80}, "chief_complaint": "chest pain"},
            correct_rec="ecg",
            wrong_rec="mri"
        )
        
        assert "recommendation" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert result["is_correct"] is True  # Easy = always correct

    def test_specialist_partial_observability(self):
        """Test specialist only sees relevant vitals."""
        specialist = SpecialistAgent(specialty="cardiology", difficulty="easy")
        
        full_state = {
            "vitals": {"heart_rate": 80, "blood_pressure": 120, "temperature": 37},
            "chief_complaint": "chest pain"
        }
        
        observed = specialist.observe(full_state)
        
        # Cardiology sees heart_rate and blood_pressure, not temperature
        assert "heart_rate" in observed["vitals"]
        assert "blood_pressure" in observed["vitals"]
        assert "temperature" not in observed["vitals"]


class TestActors:
    """Test suite for actors (Nurse, LabTech, Admin)."""

    def test_nurse_reports_vitals(self):
        """Test nurse reports vitals."""
        nurse = NurseActor()
        vitals = {"heart_rate": 80, "blood_pressure": 120}
        
        report = nurse.report_vitals(vitals)
        
        assert "heart_rate" in report
        assert "nurse_note" in report

    def test_lab_tech_processing(self):
        """Test lab tech processes tests with delays."""
        lab = LabTechActor()
        
        lab.order_test("rapid_strep_test", current_step=1, result="POSITIVE")
        lab.order_test("blood_cultures", current_step=1, result="Pending")
        
        # Rapid strep has 0 delay, should be available immediately
        available = lab.get_available(1)
        assert "rapid_strep_test" in available
        
        # Blood culture has 2 step delay
        assert "blood_cultures" not in lab.get_available(1)
        assert "blood_cultures" not in lab.get_available(2)
        assert "blood_cultures" in lab.get_available(3)

    def test_admin_approval(self):
        """Test admin approval for extra budget."""
        admin = AdminActor(initial_budget=500)
        
        # Request within budget should be approved
        result = admin.request_approval(
            test_cost=100,
            budget_remaining=200,
            clinical_justification="Patient needs test"
        )
        assert result["approved"] is True
        
        # Request over budget with good justification
        result = admin.request_approval(
            test_cost=200,
            budget_remaining=100,
            clinical_justification="This is a very important clinical justification"
        )
        assert result["approved"] is True
        assert result["extra_granted"] > 0
        
        # Request over budget with poor justification
        result = admin.request_approval(
            test_cost=200,
            budget_remaining=100,
            clinical_justification="short"
        )
        assert result["approved"] is False


class TestRewards:
    """Test suite for reward computation."""

    def test_protocol_reward_perfect(self):
        """Test perfect protocol adherence gives full reward."""
        rewards = compute_total_reward(
            tests_ordered=["request_history", "urinalysis", "urine_culture"],
            required_sequence=["request_history", "urinalysis", "urine_culture"],
            submitted_diagnosis="Urinary Tract Infection",
            correct_diagnosis="Urinary Tract Infection",
            specialist_consulted=True,
            specialist_was_correct=True,
            agent_overrode=False,
            budget_used=130,
            initial_budget=500,
            steps_used=5,
            max_steps=10,
            contraindication_triggered=False,
            patient_critical=False,
            actions_taken=["request_history", "order_test", "order_test", "consult_specialist", "submit_diagnosis"],
            reasoning="Patient symptoms indicate UTI"
        )
        
        assert rewards["protocol"] == 1.0
        assert rewards["diagnosis"] == 1.0
        assert rewards["total"] > 0.8

    def test_anti_exploit_same_test(self):
        """Test anti-exploit catches repeated same test."""
        rewards = compute_total_reward(
            tests_ordered=["test_a", "test_a", "test_a"],
            required_sequence=["test_a"],
            submitted_diagnosis="",
            correct_diagnosis="Disease",
            specialist_consulted=False,
            specialist_was_correct=False,
            agent_overrode=False,
            budget_used=300,
            initial_budget=500,
            steps_used=3,
            max_steps=10,
            contraindication_triggered=False,
            patient_critical=False,
            actions_taken=["order_test", "order_test", "order_test"],
            reasoning=""
        )
        
        assert rewards["anti_exploit"] == 0.0
        assert rewards["total"] == 0.0  # Anti-exploit zeroes total

    def test_token_efficiency(self):
        """Test token efficiency reward."""
        # Short reasoning with correct action
        rewards = compute_total_reward(
            tests_ordered=["request_history"],
            required_sequence=["request_history"],
            submitted_diagnosis="",
            correct_diagnosis="Disease",
            specialist_consulted=False,
            specialist_was_correct=False,
            agent_overrode=False,
            budget_used=0,
            initial_budget=500,
            steps_used=1,
            max_steps=10,
            contraindication_triggered=False,
            patient_critical=False,
            actions_taken=["request_history"],
            reasoning="Brief reasoning"
        )
        
        assert rewards["token_efficiency"] == 1.0


class TestCurriculum:
    """Test suite for adaptive curriculum."""

    def test_curriculum_starts_easy(self):
        """Test curriculum starts at easy."""
        curriculum = PulseCurriculum()
        stage = curriculum.get_stage()
        assert stage["name"] == "easy"

    def test_curriculum_escalates(self):
        """Test curriculum escalates when agent performs well."""
        curriculum = PulseCurriculum()
        
        # Record high rewards
        for _ in range(10):  # Exactly window of 10
            curriculum.record(0.9)
        
        stage = curriculum.get_stage()
        assert stage["name"] == "medium"

    def test_curriculum_de_escalates(self):
        """Test curriculum de-escalates when agent struggles."""
        curriculum = PulseCurriculum()
        
        # First escalate to medium
        for _ in range(10):
            curriculum.record(0.9)
        
        assert curriculum.get_stage()["name"] == "medium"
        
        # Then record poor performance
        for _ in range(15):
            curriculum.record(0.2)
        
        stage = curriculum.get_stage()
        assert stage["name"] == "easy"


class TestProtocols:
    """Test suite for protocol utilities."""

    def test_protocol_adherence_perfect(self):
        """Test perfect protocol adherence."""
        score = check_protocol_adherence(
            tests_ordered=["request_history", "urinalysis", "urine_culture"],
            required_sequence=["request_history", "urinalysis", "urine_culture"]
        )
        assert score == 1.0

    def test_protocol_adherence_partial(self):
        """Test partial protocol adherence."""
        score = check_protocol_adherence(
            tests_ordered=["request_history", "blood_test"],
            required_sequence=["request_history", "urinalysis", "urine_culture"]
        )
        assert score < 1.0
        assert score > 0.0

    def test_protocol_adherence_empty(self):
        """Test empty tests ordered."""
        score = check_protocol_adherence(
            tests_ordered=[],
            required_sequence=["request_history", "urinalysis"]
        )
        assert score == 0.0


class TestDataGenerator:
    """Test suite for data generation."""

    def test_generate_patient(self):
        """Test patient generation."""
        patient = generate_patient("uti")
        
        assert "patient_id" in patient
        assert "age" in patient
        assert "sex" in patient
        assert "vitals" in patient
        assert 18 <= patient["age"] <= 80
        assert patient["sex"] in ["male", "female"]

    def test_get_test_result(self):
        """Test test result generation."""
        result = get_test_result("uti", "urinalysis")
        assert result != ""
        assert "Positive" in result or "WBC" in result

    def test_get_test_result_unknown(self):
        """Test unknown test result."""
        result = get_test_result("uti", "unknown_test")
        assert "pending" in result.lower() or result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
