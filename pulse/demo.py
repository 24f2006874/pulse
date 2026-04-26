#!/usr/bin/env python3
"""
PULSE Demo Script
Runs pre-configured scenarios to showcase environment capabilities.
"""

import os
import sys

try:
    from pulse.server.pulse_environment import PulseEnvironment
    from pulse.models import PulseAction
except ModuleNotFoundError:
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from pulse.server.pulse_environment import PulseEnvironment
    from pulse.models import PulseAction

def demo_scenario(name, difficulty, disease, actions):
    """Run a single demo scenario."""
    print(f"\n{'='*70}")
    print(f"DEMO: {name}")
    print(f"Difficulty: {difficulty} | Disease: {disease}")
    print(f"{'='*70}")
    
    env = PulseEnvironment()
    obs = env.reset(difficulty=difficulty, disease=disease)
    
    print(f"\n📋 Initial Patient:")
    print(f"   {obs.patient_summary}")
    print(f"   Vitals: HR={obs.heart_rate:.0f} BP={obs.blood_pressure:.0f} O2={obs.oxygen_saturation:.0f}% Temp={obs.temperature:.1f}°C")
    print(f"   Available tests: {', '.join(obs.available_tests)}")
    print(f"   Diagnosis options: {', '.join(obs.diagnosis_options)}")
    
    total_reward = 0.0
    for i, action in enumerate(actions, 1):
        print(f"\n--- Step {i} ---")
        print(f"Action: {action.action_type}")
        if action.content:
            print(f"Content: {action.content}")
        if action.reasoning:
            print(f"Reasoning: {action.reasoning}")
        
        obs = env.step(action)
        total_reward += obs.reward
        
        print(f"Reward: {obs.reward:.3f} | Total: {total_reward:.3f}")
        print(f"Clinical notes: {obs.clinical_notes}")
        if obs.warning:
            print(f"⚠ Warning: {obs.warning}")
        if obs.specialist_recommendation:
            print(f"Specialist: {obs.specialist_recommendation} (conf: {obs.specialist_confidence:.0%})")
        
        if obs.done:
            print(f"\n✅ Episode ended!")
            print(f"Final reward: {obs.reward:.3f}")
            print(f"Total episode reward: {total_reward:.3f}")
            break
    
    # Print reward breakdown
    print(f"\n📊 Reward Breakdown:")
    print(f"   Protocol adherence:    {obs.reward_protocol:.3f} (28%)")
    print(f"   Diagnosis correctness: {obs.reward_diagnosis:.3f} (22%)")
    print(f"   Specialist handling:   {obs.reward_specialist:.3f} (15%)")
    print(f"   Resource efficiency:   {obs.reward_efficiency:.3f} (12%)")
    print(f"   Patient safety:        {obs.reward_safety:.3f} (8%)")
    print(f"   Anti-exploit:          {obs.reward_anti_exploit:.3f} (5%)")
    print(f"   Token efficiency:      {obs.reward_token_efficiency:.3f} (5%)")
    print(f"   Schema adaptation:     {obs.reward_schema_adaptation:.3f} (5%)")
    print(f"   TOTAL:                 {obs.reward:.3f}")
    
    return obs, total_reward


def demo_1_uti_simple():
    """Simple UTI case - easy difficulty."""
    actions = [
        PulseAction(action_type="request_history"),
        PulseAction(action_type="physical_exam"),
        PulseAction(action_type="order_test", content="urinalysis"),
        PulseAction(action_type="order_test", content="urine_culture"),
        PulseAction(action_type="submit_diagnosis", content="Urinary Tract Infection", 
                   reasoning="Classic UTI presentation with positive urinalysis and culture"),
    ]
    return demo_scenario("Simple UTI - Easy Case", "easy", "uti", actions)


def demo_2_strep_with_specialist():
    """Strep throat - specialist disagrees then corrects."""
    actions = [
        PulseAction(action_type="request_history"),
        PulseAction(action_type="physical_exam"),
        PulseAction(action_type="order_test", content="rapid_strep_test"),
        PulseAction(action_type="consult_specialist"),
        PulseAction(action_type="submit_diagnosis", content="Streptococcal Pharyngitis",
                   reasoning="Rapid strep positive, classic presentation"),
    ]
    return demo_scenario("Strep Throat - Specialist Consult", "easy", "strep_throat", actions)


def demo_3_pneumonia_budget():
    """Pneumonia - testing budget constraints."""
    actions = [
        PulseAction(action_type="request_history"),
        PulseAction(action_type="physical_exam"),
        PulseAction(action_type="order_test", content="chest_xray"),
        PulseAction(action_type="order_test", content="ct_chest", 
                   reasoning="Need better imaging"),
        PulseAction(action_type="order_test", content="blood_test"),
        PulseAction(action_type="submit_diagnosis", content="Community-Acquired Pneumonia",
                   reasoning="Chest x-ray shows consolidation"),
    ]
    return demo_scenario("Pneumonia - Budget Challenge", "medium", "pneumonia", actions)


def demo_4_dka_contraindication():
    """DKA - contraindication scenario."""
    actions = [
        PulseAction(action_type="request_history"),
        PulseAction(action_type="physical_exam"),
        PulseAction(action_type="order_test", content="blood_glucose"),
        PulseAction(action_type="order_test", content="urine_ketones"),
        PulseAction(action_type="order_test", content="blood_gas"),
        PulseAction(action_type="submit_diagnosis", content="Diabetic Ketoacidosis",
                   reasoning="Hyperglycemia, ketonuria, acidosis"),
    ]
    return demo_scenario("DKA - Protocol Following", "medium", "dka", actions)


def demo_5_sepsis_long_horizon():
    """Sepsis - long horizon case."""
    actions = [
        PulseAction(action_type="request_history"),
        PulseAction(action_type="physical_exam"),
        PulseAction(action_type="order_test", content="blood_cultures"),
        PulseAction(action_type="order_test", content="lactate_level"),
        PulseAction(action_type="order_test", content="blood_test"),
        PulseAction(action_type="consult_specialist"),
        PulseAction(action_type="submit_diagnosis", content="Sepsis",
                   reasoning="Hypotension, confusion, elevated lactate, likely urosepsis"),
    ]
    return demo_scenario("Sepsis - Hard Case", "hard", "sepsis", actions)


def demo_6_schema_drift():
    """Demonstrate schema drift (may need multiple runs)."""
    print(f"\n{'='*70}")
    print(f"DEMO: Schema Drift (Patronus AI Bonus)")
    print(f"{'='*70}")
    print("Note: Schema drift has 10% chance at steps 4, 5, 6")
    print("Running pneumonia case (drift affects pneumonia) ...\n")
    
    env = PulseEnvironment()
    obs = env.reset(difficulty="medium", disease="pneumonia")
    
    actions = [
        PulseAction(action_type="request_history"),
        PulseAction(action_type="physical_exam"),
        PulseAction(action_type="order_test", content="chest_xray"),
        # Step 4 - drift may occur here if pneumonia + 10% roll
        PulseAction(action_type="order_test", content="blood_test"),
        PulseAction(action_type="acknowledge_schema_change"),
        PulseAction(action_type="submit_diagnosis", content="Community-Acquired Pneumonia",
                   reasoning="Following updated protocol"),
    ]
    
    total_reward = 0.0
    for i, action in enumerate(actions, 1):
        print(f"\nStep {i}: {action.action_type}")
        obs = env.step(action)
        total_reward += obs.reward
        
        if obs.schema_alert:
            print(f"🚨 SCHEMA DRIFT DETECTED: {obs.schema_alert}")
            print(f"   Type: {obs.schema_change_type}")
        
        print(f"Reward: {obs.reward:.3f}")
        
        if obs.done:
            break
    
    if obs.reward_schema_adaptation == 1.0:
        print(f"\n✅ Agent successfully adapted to schema drift!")
    elif obs.reward_schema_adaptation == 0.0:
        print(f"\n❌ Agent did not adapt to schema drift")
    
    return obs, total_reward


def demo_baseline_random():
    """Baseline: Random agent."""
    import random
    
    print(f"\n{'='*70}")
    print(f"BASELINE: Random Agent")
    print(f"{'='*70}")
    
    env = PulseEnvironment()
    obs = env.reset(difficulty="easy", disease="uti")
    
    valid_actions = [
        "request_history", "physical_exam", "order_test",
        "consult_specialist", "submit_diagnosis", "wait"
    ]
    
    tests = ["urinalysis", "urine_culture", "blood_test"]
    diagnoses = [
        "Urinary Tract Infection", "Kidney Stone",
        "Pelvic Inflammatory Disease", "Interstitial Cystitis",
        "Vaginal Infection"
    ]
    
    total_reward = 0.0
    for step in range(1, 7):  # UTI max_steps=6
        if obs.done:
            break
        
        action_type = random.choice(valid_actions)
        content = ""
        reasoning = "Random choice"
        
        if action_type == "order_test":
            content = random.choice(tests)
        elif action_type == "submit_diagnosis":
            content = random.choice(diagnoses)
        
        action = PulseAction(action_type=action_type, content=content, reasoning=reasoning)
        obs = env.step(action)
        total_reward += obs.reward
        
        print(f"Step {step}: {action_type:20s} | Reward: {obs.reward:6.3f} | Total: {total_reward:6.3f}")
    
    print(f"\nFinal reward: {total_reward:.3f}")
    return obs, total_reward


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PULSE ENVIRONMENT - DEMO SUITE")
    print("="*70)
    
    # Run demos
    demo_1_uti_simple()
    demo_2_strep_with_specialist()
    demo_3_pneumonia_budget()
    demo_4_dka_contraindication()
    demo_5_sepsis_long_horizon()
    
    # Schema drift (may or may not trigger - 10% chance)
    for attempt in range(3):
        obs, _ = demo_6_schema_drift()
        if obs.reward_schema_adaptation < 1.0:
            # Drift occurred and agent adapted
            break
        if attempt < 2:
            print(f"\n(Schema drift didn't trigger, retrying...)\n")
    
    # Baseline comparison
    demo_baseline_random()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")
