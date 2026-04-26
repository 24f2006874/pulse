# inference.py
# MANDATORY at project root
# Exact [START][STEP][END] format

import asyncio
import os
import sys
import json
from typing import List, Optional
from openai import AsyncOpenAI

try:
    from pulse.models import PulseAction
    from pulse.server.pulse_environment import PulseEnvironment
except ModuleNotFoundError:
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from pulse.models import PulseAction
    from pulse.server.pulse_environment import PulseEnvironment

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://router.huggingface.ai/hf-inference/v1"
)
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "meta-llama/Llama-3.3-70B-Instruct"
)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MAX_STEPS = 15
MAX_TOTAL_REWARD = 10.0
SUCCESS_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are PULSE - a clinical protocol AI.

Your task: Execute the correct sequence of clinical actions
to diagnose and manage the patient.

ACTIONS (respond in JSON):
{"action_type": "request_history", "content": "", "reasoning": "Need patient history first"}
{"action_type": "physical_exam", "content": "", "reasoning": "Assess vitals"}
{"action_type": "order_test", "content": "test_name", "reasoning": "Why this test"}
{"action_type": "administer_treatment", "content": "treatment_name", "reasoning": "Why this treatment now"}
{"action_type": "consult_specialist", "content": "", "reasoning": "Need expert input"}
{"action_type": "override_specialist", "content": "", "reasoning": "Evidence contradicts specialist"}
{"action_type": "submit_diagnosis", "content": "Exact diagnosis from options", "reasoning": "Evidence supports"}

RULES:
- Follow correct protocol sequence
- Consult specialist before diagnosing
- Use evidence to decide when to override
- Be concise in reasoning (under 50 words)
- Submit diagnosis EXACTLY from options provided"""


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True
    )


def build_prompt(obs) -> str:
    return f"""
PATIENT: {getattr(obs, 'patient_summary', '')}
VITALS: HR={getattr(obs, 'heart_rate', 80):.0f} BP={getattr(obs, 'blood_pressure', 120):.0f} O2={getattr(obs, 'oxygen_saturation', 98):.0f}%
NOTES: {getattr(obs, 'clinical_notes', '')}
TESTS AVAILABLE: {getattr(obs, 'available_tests', [])}
TEST RESULTS: {getattr(obs, 'test_results', {})}
SPECIALIST SAYS: {getattr(obs, 'specialist_recommendation', 'Not consulted')}
SPECIALIST UPDATED: {getattr(obs, 'specialist_updated', False)}
UPDATE REASON: {getattr(obs, 'specialist_update_reason', '')}
NURSE REPORT: {getattr(obs, 'nurse_vitals_report', {})}
SCHEMA ALERT: {getattr(obs, 'schema_alert', '')}
DIAGNOSIS OPTIONS: {getattr(obs, 'diagnosis_options', [])}
BUDGET REMAINING: ${getattr(obs, 'budget_remaining', 500):.0f}
STEPS REMAINING: {getattr(obs, 'steps_remaining', 15)}
WARNING: {getattr(obs, 'warning', '')}

What is your next clinical action? Respond in JSON only.
"""


async def run_episode(
    client: AsyncOpenAI,
    difficulty: str,
) -> float:
    env = PulseEnvironment()

    log_start(
        task=f"clinical_protocol_{difficulty}",
        env="pulse",
        model=MODEL_NAME
    )

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        obs = env.reset(difficulty=difficulty)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            messages.append({
                "role": "user",
                "content": build_prompt(obs)
            })

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.3,
            )

            text = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": text})

            try:
                data = json.loads(text)
                action = PulseAction(
                    action_type=data.get("action_type", "wait"),
                    content=data.get("content", ""),
                    reasoning=data.get("reasoning", ""),
                )
            except Exception:
                action = PulseAction(action_type="wait")

            obs = env.step(action)

            reward = obs.reward or 0.0
            done = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action.action_type,
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


async def main():
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Export a Hugging Face token before running inference.py."
        )

    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    difficulties = ["easy", "medium", "hard"]
    total = 0.0

    for diff in difficulties:
        score = await run_episode(client, diff)
        total += score
        print(f"[SUMMARY] difficulty={diff} score={score:.4f}", flush=True)

    avg = total / len(difficulties)
    print(f"[FINAL] average_score={avg:.4f}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", flush=True)
        raise
