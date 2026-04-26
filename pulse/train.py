#!/usr/bin/env python3
"""Reproducible Unsloth GRPO training for PULSE.

Key properties:
- Uses the real PulseEnvironment reward (no helper-action leakage).
- Uses all diseases in the protocol catalog for prompt generation.
- Writes an evaluation artifact after training for README/reporting.

Required packages:
    pip install unsloth trl transformers datasets wandb

Environment variables:
    HF_TOKEN        Hugging Face token (required for gated/base model access)
    WANDB_API_KEY   Optional W&B key. If omitted, W&B is disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

try:
    from pulse.models import PulseAction
    from pulse.server.protocols import PROTOCOLS
    from pulse.server.pulse_environment import PulseEnvironment
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from pulse.models import PulseAction
    from pulse.server.protocols import PROTOCOLS
    from pulse.server.pulse_environment import PulseEnvironment

try:
    import wandb
except Exception:
    wandb = None


SYSTEM_PROMPT = """You are a clinical protocol AI.
Respond with JSON only in exactly this format:
{"action_type":"...","content":"...","reasoning":"..."}

Allowed action_type values:
request_history, physical_exam, order_test, administer_treatment,
consult_specialist, override_specialist, request_admin_approval,
acknowledge_schema_change, submit_diagnosis, wait
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PULSE policy with GRPO")
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", default="./pulse-trained")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--dataset-repeats", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def configure_auth(disable_wandb: bool) -> bool:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to run training.")

    use_wandb = False
    if not disable_wandb and wandb is not None:
        wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
        if wandb_key:
            wandb.login(key=wandb_key)
            use_wandb = True
    return use_wandb


def build_prompt(disease: str) -> str:
    protocol = PROTOCOLS[disease]
    tests = list(protocol.get("available_tests", {}).keys())
    diagnoses = protocol.get("diagnosis_choices", [])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Patient: {protocol.get('presentation', '')}\n"
        f"Available tests: {tests}\n"
        f"Diagnosis options: {diagnoses}\n"
        f"Difficulty: {protocol.get('difficulty', 'easy')}\n"
        "JSON:"
    )


def build_dataset(repeats: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for disease in PROTOCOLS.keys():
        for _ in range(repeats):
            rows.append(
                {
                    "prompt": build_prompt(disease),
                    "disease": disease,
                    "difficulty": PROTOCOLS[disease].get("difficulty", "easy"),
                }
            )

    rng.shuffle(rows)
    return Dataset.from_list(rows)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "completion"):
            if key in completion:
                return str(completion[key])
        return str(completion)
    if isinstance(completion, (list, tuple)) and completion:
        return completion_to_text(completion[0])
    return str(completion)


def parse_action(raw_text: str) -> dict[str, str]:
    text = (raw_text or "").strip()

    try:
        start = text.find("{")
        if start >= 0:
            depth = 0
            end = -1
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                parsed = json.loads(text[start:end])
                return {
                    "action_type": str(parsed.get("action_type", "wait")),
                    "content": str(parsed.get("content", "")),
                    "reasoning": str(parsed.get("reasoning", "")),
                }
    except Exception:
        pass

    lower = text.lower()
    for action_type in (
        "submit_diagnosis",
        "request_history",
        "physical_exam",
        "order_test",
        "administer_treatment",
        "consult_specialist",
        "override_specialist",
        "request_admin_approval",
        "acknowledge_schema_change",
        "wait",
    ):
        if action_type in lower:
            return {
                "action_type": action_type,
                "content": "",
                "reasoning": text[:120],
            }

    return {"action_type": "wait", "content": "", "reasoning": "fallback"}


def disease_from_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for disease, protocol in PROTOCOLS.items():
        presentation = protocol.get("presentation", "").lower()
        if presentation and presentation in prompt_lower:
            return disease
    return "uti"


def grpo_reward_function(prompts, completions, disease=None, difficulty=None, **kwargs):
    rewards: list[float] = []
    components: dict[str, list[float]] = {
        "protocol": [],
        "diagnosis": [],
        "efficiency": [],
        "safety": [],
        "token_efficiency": [],
        "schema_adaptation": [],
    }

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            d = disease[i] if disease is not None else disease_from_prompt(str(prompt))
            if d not in PROTOCOLS:
                d = "uti"

            diff = (
                difficulty[i]
                if difficulty is not None
                else PROTOCOLS[d].get("difficulty", "easy")
            )

            env = PulseEnvironment()
            env.reset(difficulty=diff, disease=d, seed=10_000 + i)

            text = completion_to_text(completion)
            action = parse_action(text)
            obs = env.step(
                PulseAction(
                    action_type=action["action_type"],
                    content=action["content"],
                    reasoning=action["reasoning"],
                )
            )

            reward = float(getattr(obs, "reward", 0.0) or 0.0)
            rewards.append(reward)

            components["protocol"].append(float(getattr(obs, "reward_protocol", 0.0)))
            components["diagnosis"].append(float(getattr(obs, "reward_diagnosis", 0.0)))
            components["efficiency"].append(float(getattr(obs, "reward_efficiency", 0.0)))
            components["safety"].append(float(getattr(obs, "reward_safety", 0.0)))
            components["token_efficiency"].append(
                float(getattr(obs, "reward_token_efficiency", 0.0))
            )
            components["schema_adaptation"].append(
                float(getattr(obs, "reward_schema_adaptation", 0.0))
            )
        except Exception:
            rewards.append(0.0)

    if wandb is not None and wandb.run is not None and rewards:
        wandb.log(
            {
                "reward/total_batch_mean": mean(rewards),
                "reward/protocol_batch_mean": mean(components["protocol"]),
                "reward/diagnosis_batch_mean": mean(components["diagnosis"]),
                "reward/efficiency_batch_mean": mean(components["efficiency"]),
                "reward/safety_batch_mean": mean(components["safety"]),
                "reward/token_efficiency_batch_mean": mean(
                    components["token_efficiency"]
                ),
                "reward/schema_adaptation_batch_mean": mean(
                    components["schema_adaptation"]
                ),
            }
        )

    return rewards


def build_model(model_name: str, max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    return model, tokenizer


def init_trainer(model, tokenizer, train_dataset, args):
    try:
        return GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=[grpo_reward_function],
            args=args,
        )
    except TypeError:
        return GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            reward_function=grpo_reward_function,
            args=args,
        )


def generate_action(model, tokenizer, prompt: str) -> dict[str, str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    completion = text[len(prompt) :] if text.startswith(prompt) else text
    return parse_action(completion)


def evaluate_policy(model, tokenizer, episodes: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    totals = []
    protocols = []
    diagnosis = []

    for i in range(episodes):
        disease = rng.choice(list(PROTOCOLS.keys()))
        difficulty = PROTOCOLS[disease].get("difficulty", "easy")

        env = PulseEnvironment()
        obs = env.reset(difficulty=difficulty, disease=disease, seed=seed + i)

        # Short rollout for fast sanity checking after training.
        for _ in range(min(5, PROTOCOLS[disease].get("max_steps", 10))):
            if obs.done:
                break

            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Patient: {obs.patient_summary}\n"
                f"Available tests: {obs.available_tests}\n"
                f"Diagnosis options: {obs.diagnosis_options}\n"
                f"Clinical notes: {obs.clinical_notes}\n"
                f"Warning: {obs.warning}\n"
                "JSON:"
            )

            action = generate_action(model, tokenizer, prompt)
            obs = env.step(
                PulseAction(
                    action_type=action["action_type"],
                    content=action["content"],
                    reasoning=action["reasoning"],
                )
            )

        totals.append(float(obs.reward))
        protocols.append(float(getattr(obs, "reward_protocol", 0.0)))
        diagnosis.append(float(getattr(obs, "reward_diagnosis", 0.0)))

    return {
        "episodes": float(episodes),
        "mean_total_reward": float(mean(totals) if totals else 0.0),
        "mean_protocol_reward": float(mean(protocols) if protocols else 0.0),
        "mean_diagnosis_reward": float(mean(diagnosis) if diagnosis else 0.0),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    has_wandb = configure_auth(args.disable_wandb)

    if has_wandb:
        wandb.init(
            project="pulse-clinical-ai",
            name="pulse-grpo-clean-reward",
            config={
                "model_name": args.model_name,
                "dataset_repeats": args.dataset_repeats,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "reward_mode": "single_step_real_env_no_helper_actions",
            },
        )

    dataset = build_dataset(args.dataset_repeats, args.seed)
    print(f"Dataset size: {len(dataset)}")

    print("Loading model...")
    model, tokenizer = build_model(args.model_name, args.max_seq_length)

    training_args = GRPOConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=args.seed,
        output_dir=args.output_dir,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_prompt_length=768,
        max_completion_length=80,
        report_to="wandb" if has_wandb else "none",
        run_name="pulse-grpo-clean-reward",
    )

    trainer = init_trainer(model, tokenizer, dataset, training_args)

    print("=" * 48)
    print("GRPO TRAINING STARTING")
    if has_wandb and wandb.run is not None:
        print(f"W&B run: {wandb.run.get_url()}")
    print("=" * 48)

    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Running post-train evaluation...")
    metrics = evaluate_policy(model, tokenizer, args.eval_episodes, args.seed + 1)
    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {metrics_path}")

    if has_wandb and wandb.run is not None:
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    main()
