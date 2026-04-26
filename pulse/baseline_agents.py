#!/usr/bin/env python3
"""
Baseline Agents for PULSE Environment
Provides comparison baselines for RL agent performance.
"""

import os
import random
import sys
from typing import List, Optional, Dict

try:
    from pulse.models import PulseAction
    from pulse.server.pulse_environment import PulseEnvironment
except ModuleNotFoundError:
    from models import PulseAction
    from server.pulse_environment import PulseEnvironment


class BaselineAgent:
    """Base class for all baseline agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset agent state."""
        pass
    
    def select_action(self, obs) -> PulseAction:
        """Select action given observation.
        
        Args:
            obs: PulseObservation from environment
            
        Returns:
            PulseAction to take
        """
        raise NotImplementedError


class RandomAgent(BaselineAgent):
    """Random agent - selects actions randomly."""
    
    def __init__(self):
        super().__init__("Random Agent")
        self.valid_actions = [
            "request_history", "physical_exam", "order_test",
            "consult_specialist", "submit_diagnosis", "wait",
            "acknowledge_schema_change"
        ]
    
    def reset(self):
        super().reset()
    
    def select_action(self, obs) -> PulseAction:
        action_type = random.choice(self.valid_actions)
        
        content = ""
        reasoning = "Random selection"
        
        if action_type == "order_test" and obs.available_tests:
            content = random.choice(obs.available_tests)
        elif action_type == "submit_diagnosis" and obs.diagnosis_options:
            content = random.choice(obs.diagnosis_options)
        
        return PulseAction(
            action_type=action_type,
            content=content,
            reasoning=reasoning
        )


class GreedyOrderAgent(BaselineAgent):
    """Greedy agent - orders all available tests, then diagnoses."""
    
    def __init__(self):
        super().__init__("Greedy Order Agent")
        self.ordered_tests = set()
        self.has_consulted = False
        self.has_diagnosed = False
    
    def reset(self):
        super().reset()
        self.ordered_tests = set()
        self.has_consulted = False
        self.has_diagnosed = False
    
    def select_action(self, obs) -> PulseAction:
        # If not diagnosed yet
        if not self.has_diagnosed:
            # Check if there are unordered tests
            available = set(obs.available_tests or [])
            unordered = available - self.ordered_tests
            
            if unordered and len(self.ordered_tests) < 5:
                # Order another test
                test = random.choice(list(unordered))
                self.ordered_tests.add(test)
                return PulseAction(
                    action_type="order_test",
                    content=test,
                    reasoning=f"Ordering {test} to gather information"
                )
            
            # Consult specialist if not done
            if not self.has_consulted and not obs.specialist_recommendation:
                self.has_consulted = True
                return PulseAction(
                    action_type="consult_specialist",
                    reasoning="Need expert opinion before diagnosis"
                )
            
            # Submit diagnosis
            if obs.diagnosis_options:
                # If specialist gave recommendation, follow it
                if obs.specialist_recommendation and obs.specialist_recommendation in obs.diagnosis_options:
                    diagnosis = obs.specialist_recommendation
                    reasoning = "Following specialist recommendation"
                else:
                    # Otherwise pick first option (or random)
                    diagnosis = obs.diagnosis_options[0]
                    reasoning = "Best guess based on available information"
                
                self.has_diagnosed = True
                return PulseAction(
                    action_type="submit_diagnosis",
                    content=diagnosis,
                    reasoning=reasoning
                )
        
        # Wait as fallback
        return PulseAction(
            action_type="wait",
            reasoning="No better action available"
        )


class ProtocolFollowingAgent(BaselineAgent):
    """Rule-based agent that follows known protocols.
    
    This is a simplified rule-based agent that attempts to follow
    the correct protocol sequence when it can infer the disease
    from available information.
    """
    
    def __init__(self):
        super().__init__("Protocol Following Agent")
        from server.protocols import PROTOCOLS
        self.protocols = PROTOCOLS
        self.current_protocol = None
        self.protocol_step = 0
        self.consulted_specialist = False
        self.has_requested_history = False
        self.has_examined = False
        self.has_ordered_test = False
        self.has_consulted = False
    
    def reset(self):
        super().reset()
        self.current_protocol = None
        self.protocol_step = 0
        self.consulted_specialist = False
        self.has_requested_history = False
        self.has_examined = False
        self.has_ordered_test = False
        self.has_consulted = False
    
    def _infer_disease(self, obs) -> Optional[str]:
        """Attempt to infer disease from observation."""
        # Check if disease is obvious from presentation or test results
        presentation = getattr(obs, 'patient_summary', '')
        test_results = getattr(obs, 'test_results', {})
        
        # Check test results for strong indicators
        for test, result in test_results.items():
            if 'Positive' in result and 'Strep' in result:
                return 'strep_throat'
            if 'Glucose' in test and '480' in result:
                return 'dka'
            if 'Lactate' in test and 'elevated' in result.lower():
                return 'sepsis'
            if 'consolidation' in result.lower():
                return 'pneumonia'
            if 'E. coli' in result:
                return 'uti'
        
        # Check presentation keywords
        if 'Burning urination' in presentation or 'UTI' in presentation:
            return 'uti'
        if 'sore throat' in presentation.lower():
            return 'strep_throat'
        if 'diabetic' in presentation.lower() or 'DKA' in presentation:
            return 'dka'
        if 'pneumonia' in presentation.lower() or 'consolidation' in presentation:
            return 'pneumonia'
        if 'sepsis' in presentation.lower() or 'Confusion' in presentation:
            return 'sepsis'
        
        return None
    
    def select_action(self, obs) -> PulseAction:
        # Infer disease if not already known
        if self.current_protocol is None:
            disease = self._infer_disease(obs)
            if disease and disease in self.protocols:
                self.current_protocol = self.protocols[disease]
        
        # If we have a protocol, follow it
        if self.current_protocol:
            required = self.current_protocol.get('required_sequence', [])
            
            # Follow protocol sequence
            if self.protocol_step < len(required):
                next_action = required[self.protocol_step]
                
                if next_action == 'request_history':
                    self.protocol_step += 1
                    return PulseAction(
                        action_type='request_history',
                        reasoning='Following protocol: get patient history'
                    )
                elif next_action == 'physical_exam':
                    self.protocol_step += 1
                    return PulseAction(
                        action_type='physical_exam',
                        reasoning='Following protocol: physical examination'
                    )
                elif next_action in (obs.available_tests or []):
                    self.protocol_step += 1
                    return PulseAction(
                        action_type='order_test',
                        content=next_action,
                        reasoning=f'Following protocol: order {next_action}'
                    )
            
            # After completing protocol, consult specialist if not done
            if not self.consulted_specialist and not obs.specialist_recommendation:
                self.consulted_specialist = True
                return PulseAction(
                    action_type='consult_specialist',
                    reasoning='Protocol complete, consulting specialist'
                )
            
            # Submit diagnosis
            if obs.diagnosis_options:
                correct_dx = self.current_protocol.get('correct_diagnosis', '')
                if correct_dx in obs.diagnosis_options:
                    diagnosis = correct_dx
                    reasoning = 'Following protocol diagnosis'
                elif obs.specialist_recommendation in obs.diagnosis_options:
                    diagnosis = obs.specialist_recommendation
                    reasoning = 'Following specialist recommendation'
                else:
                    diagnosis = obs.diagnosis_options[0]
                    reasoning = 'Best available option'
                
                return PulseAction(
                    action_type='submit_diagnosis',
                    content=diagnosis,
                    reasoning=reasoning
                )
        
        # Fallback: if no protocol or unknown disease, use heuristic
        if not self.has_requested_history:
            self.has_requested_history = True
            return PulseAction(
                action_type='request_history',
                reasoning='Need to understand patient presentation'
            )
        
        if not self.has_examined:
            self.has_examined = True
            return PulseAction(
                action_type='physical_exam',
                reasoning='Need to assess patient'
            )
        
        # Order first available test
        if obs.available_tests and not self.has_ordered_test:
            self.has_ordered_test = True
            return PulseAction(
                action_type='order_test',
                content=obs.available_tests[0],
                reasoning='Need diagnostic information'
            )
        
        # Consult specialist as last resort
        if not self.has_consulted:
            self.has_consulted = True
            return PulseAction(
                action_type='consult_specialist',
                reasoning='Need expert guidance'
            )
        
        # Submit diagnosis if options available
        if obs.diagnosis_options:
            return PulseAction(
                action_type='submit_diagnosis',
                content=obs.diagnosis_options[0],
                reasoning='Making best guess'
            )
        
        return PulseAction(
            action_type='wait',
            reasoning='No action available'
        )


class ConservativeAgent(BaselineAgent):
    """Conservative agent - prioritizes safety and thoroughness.
    
    Orders all tests, consults specialist, then diagnoses.
    Avoids budget overruns and contraindications.
    """
    
    def __init__(self):
        super().__init__("Conservative Agent")
        self.ordered_tests = []
        self.consulted = False
        self.budget_threshold = 0.8  # Don't exceed 80% of budget
    
    def reset(self):
        super().reset()
        self.ordered_tests = []
        self.consulted = False
    
    def select_action(self, obs) -> PulseAction:
        # Check budget
        budget_pct = 0
        if obs.budget_remaining > 0:
            # Approximate budget usage
            budget_pct = 1.0 - (obs.budget_remaining / 500.0)
        
        # Phase 1: Order tests (but not too many)
        if len(self.ordered_tests) < 3 and budget_pct < self.budget_threshold:
            available = [t for t in (obs.available_tests or []) 
                        if t not in self.ordered_tests]
            if available:
                test = available[0]  # Pick first available
                self.ordered_tests.append(test)
                return PulseAction(
                    action_type='order_test',
                    content=test,
                    reasoning=f'Conservative approach: order {test}'
                )
        
        # Phase 2: Consult specialist
        if not self.consulted:
            self.consulted = True
            return PulseAction(
                action_type='consult_specialist',
                reasoning='Conservative: get expert opinion before acting'
            )
        
        # Phase 3: Review specialist recommendation
        if obs.specialist_recommendation:
            # If specialist says something specific, consider following
            if obs.specialist_confidence > 0.8:
                # High confidence - follow recommendation
                if obs.specialist_recommendation in (obs.available_tests or []):
                    if obs.specialist_recommendation not in self.ordered_tests:
                        self.ordered_tests.append(obs.specialist_recommendation)
                        return PulseAction(
                            action_type='order_test',
                            content=obs.specialist_recommendation,
                            reasoning=f'Specialist confident ({obs.specialist_confidence:.0%}), following recommendation'
                        )
        
        # Phase 4: Make diagnosis if we have enough information
        if len(self.ordered_tests) >= 2 and self.consulted:
            if obs.diagnosis_options:
                # If specialist was consulted and gave opinion, consider it
                if obs.specialist_recommendation and obs.specialist_recommendation in obs.diagnosis_options:
                    diagnosis = obs.specialist_recommendation
                    reasoning = 'Following specialist recommendation'
                else:
                    diagnosis = obs.diagnosis_options[0]
                    reasoning = 'Conservative: diagnosis after thorough evaluation'
                
                return PulseAction(
                    action_type='submit_diagnosis',
                    content=diagnosis,
                    reasoning=reasoning
                )
        
        # Wait if not ready
        return PulseAction(
            action_type='wait',
            reasoning='Conservative: waiting for more information'
        )


def run_baseline_comparison(difficulty='easy', disease='uti', num_episodes=5):
    """Run comparison of all baseline agents."""
    from pulse.server.pulse_environment import PulseEnvironment
    
    agents = [
        RandomAgent(),
        GreedyOrderAgent(),
        ProtocolFollowingAgent(),
        ConservativeAgent()
    ]
    
    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISON")
    print(f"Difficulty: {difficulty} | Disease: {disease}")
    print(f"Episodes per agent: {num_episodes}")
    print(f"{'='*70}\n")
    
    results = {}
    
    for agent in agents:
        print(f"Running {agent.name}...")
        scores = []
        
        for ep in range(num_episodes):
            env = PulseEnvironment()
            obs = env.reset(difficulty=difficulty, disease=disease, seed=ep*42)
            
            agent.reset()
            total_reward = 0.0
            steps = 0
            
            while not obs.done and steps < 15:
                action = agent.select_action(obs)
                obs = env.step(action)
                total_reward += obs.reward
                steps += 1
            
            scores.append(total_reward)
        
        avg_score = sum(scores) / len(scores)
        results[agent.name] = {
            'scores': scores,
            'average': avg_score,
            'min': min(scores),
            'max': max(scores)
        }
        
        print(f"  Average: {avg_score:.3f} | Min: {min(scores):.3f} | Max: {max(scores):.3f}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Agent':<30} {'Avg Score':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*60}")
    
    for name, data in results.items():
        print(f"{name:<30} {data['average']:>10.3f} {data['min']:>10.3f} {data['max']:>10.3f}")
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['average'])
    print(f"\n🏆 Best performing agent: {best_name}")
    print(f"   Average score: {results[best_name]['average']:.3f}\n")
    
    return results


if __name__ == '__main__':
    run_baseline_comparison(difficulty='easy', disease='uti', num_episodes=5)
