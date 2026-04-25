---
title: PULSE Environment Server
emoji: "🩺"
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - healthcare
---

# PULSE

PULSE is an OpenEnv-compatible clinical protocol reinforcement learning environment. An agent must manage a synthetic patient case across multiple steps, choose the right diagnostic actions, react to delayed results and specialist input, and optimize for both clinical correctness and operational efficiency.

This project is built as a benchmark-style environment for long-horizon reasoning rather than a toy echo demo. The current repo includes:

- a FastAPI/OpenEnv server
- a Python client
- synthetic patient generation
- protocol-grounded rewards
- multi-actor dynamics
- curriculum difficulty scaling
- baseline agents
- a demo script
- a passing test suite

## What The Agent Does

At each step, the agent selects a structured action:

- `request_history`
- `physical_exam`
- `order_test`
- `consult_specialist`
- `override_specialist`
- `request_admin_approval`
- `acknowledge_schema_change`
- `submit_diagnosis`
- `wait`

The environment returns a structured observation containing:

- patient summary
- available tests
- test results
- diagnosis options
- specialist recommendation and confidence
- nurse and lab information
- schema drift alerts
- vitals and budget
- reward breakdown

## Why This Environment Is Interesting

PULSE is designed to test more than single-step classification.

- Long-horizon reasoning: agents must follow multi-step protocols, not just guess diagnoses.
- Multi-actor dynamics: specialists, nurses, lab delays, and admin approval affect decisions.
- Reward transparency: rewards are fully programmatic and decomposed into interpretable components.
- Resource constraints: agents must manage budget, steps, and patient deterioration.
- Distribution shift: schema drift events force adaptation mid-episode.

## Project Structure

```text
pulse/
├── __init__.py
├── baseline_agents.py
├── client.py
├── demo.py
├── Dockerfile
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── server/
│   ├── __init__.py
│   ├── actors.py
│   ├── app.py
│   ├── curriculum.py
│   ├── data_generator.py
│   ├── protocols.py
│   ├── pulse_environment.py
│   ├── requirements.txt
│   ├── rewards.py
│   └── specialist.py
└── tests/
    └── test_environment.py
```

## Core Components

`models.py`

- Defines `PulseAction`, `PulseObservation`, and `PulseState`.

`server/pulse_environment.py`

- Main environment engine.
- Handles reset, step, deterioration, schema drift, reward calculation, and actor interaction.

`server/protocols.py`

- Hard-coded disease protocols and diagnosis options.

`server/rewards.py`

- Eight reward components:
- protocol adherence
- diagnosis correctness
- specialist handling
- efficiency
- safety
- anti-exploit
- token efficiency
- schema adaptation

`server/actors.py`

- Nurse, lab technician, and admin actors.

`server/specialist.py`

- Specialist recommendation logic with partial observability and mid-episode updates.

`baseline_agents.py`

- Baseline agents for comparison, including random, greedy, and protocol-following behavior.

`demo.py`

- Pre-configured scenarios for qualitative demos.

## Quick Start

### Local Python Run

From the project root:

```powershell
python -m pytest tests\test_environment.py -q
python server\app.py
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`
- `http://localhost:8000/web`

Notes:

- `/docs` is the safest route for direct API inspection.
- `/web` is available through the OpenEnv UI layer and is the easiest way to interact manually.
- `/` may return `404`, which is expected because no homepage route is defined.

### Docker Run

Build and run from the project root:

```powershell
docker build --no-cache -t pulse .
docker run --rm -p 8000:8000 pulse
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`
- `http://localhost:8000/web`

## Python Client Example

```python
from pulse import PulseAction
from pulse.client import PulseEnv

with PulseEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print(obs.observation.patient_summary)

    result = env.step(
        PulseAction(
            action_type="request_history",
            reasoning="Need patient history first",
        )
    )
    print(result.observation.clinical_notes)
```

## Running The Demo

```powershell
python demo.py
```

The demo showcases:

- easy UTI flow
- strep throat with specialist interaction
- pneumonia with budget pressure
- DKA protocol flow
- sepsis long-horizon reasoning
- schema drift behavior
- random baseline behavior

## Running Tests

```powershell
python -m pytest tests\test_environment.py -q
```

Current verified baseline:

- `42 passed`

## Running Inference

`inference.py` runs an LLM-driven policy loop against the local PULSE environment logic.

Required environment variables:

```powershell
$env:HF_TOKEN="your_huggingface_token"
$env:API_BASE_URL="https://router.huggingface.ai/hf-inference/v1"
$env:MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
```

Then run:

```powershell
python inference.py
```

If `HF_TOKEN` is missing, the script now fails fast with a clear error message.

## Hugging Face / OpenEnv Deployment

The repo includes `openenv.yaml` and a Dockerfile for OpenEnv-style deployment.

Typical command:

```powershell
openenv push
```

After deployment, verify:

- `/docs`
- `/health`
- `/web`

## Current Strengths

- OpenEnv-compatible environment loop
- structured action and observation schema
- multi-step clinical reasoning task
- multi-actor environment behavior
- reward shaping with explicit components
- curriculum support
- baseline agents and demo support
- passing test suite

## Current Known Rough Edges

- The project is stronger as a benchmark/simulation than as a polished product.
- Some environment dynamics are benchmark-oriented rather than medically realistic.
- Browser console warnings for `favicon.ico` or `manifest.json` are cosmetic if `/web` works.
- Packaging metadata can still be tightened further if the project is prepared for wider distribution.

## Hackathon Positioning

PULSE is strongest when presented as:

- a benchmark for protocol-following agents
- a testbed for long-horizon reasoning under constraints
- a transparent, judgeable RL environment with structured rewards

The project’s biggest upgrade path is clearer presentation:

- a polished README
- benchmark results for baseline agents
- a short demo flow for judges
- a crisp explanation of why this environment matters
