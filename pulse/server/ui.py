import gradio as gr
import os
import sys

try:
    from .pulse_environment import PulseEnvironment
    from pulse.models import PulseAction
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from pulse.server.pulse_environment import PulseEnvironment
    from pulse.models import PulseAction

# Global instance
env = PulseEnvironment()
current_obs = None
current_difficulty = "easy"

UI_THEME = gr.themes.Soft()
UI_CSS = """
.pulse-header { background: linear-gradient(135deg, #0f4c81, #0a6b5f); color:white; padding:20px; border-radius:12px; text-align:center; }
.vital-critical { color: #ff3333; font-weight:bold; }
.container { border: 2px solid #0f4c81; border-radius: 10px; padding: 15px; }
"""

def format_health_bar(o2, bp, hr, temp):
    health = max(0, min(100, int((o2 - 70) / 30 * 100)))
    color = "🟢" if health > 70 else "🟡" if health > 40 else "🔴"
    bar = "█" * (health // 5) + "░" * (20 - (health // 5))
    return f"{color} HEALTH [{bar}] {health}%"

def update_ui(obs):
    if obs is None:
        return (
            "No patient loaded. Click Reset.",
            "Vitals not available",
            "No notes yet",
            "Specialist not consulted",
            "No test results",
            "Reward: 0.00",
            "Episode not started"
        )

    patient_info = f"""
**Patient ID:** {getattr(obs, 'patient_id', 'Unknown')}
**Age/Sex:** {getattr(obs, 'age', '?')}yo {getattr(obs, 'sex', '?')}
**Chief Complaint:** {obs.patient_summary[:150]}...
"""

    vitals = f"""
{format_health_bar(
    getattr(obs, 'oxygen_saturation', 98),
    getattr(obs, 'blood_pressure', 120),
    getattr(obs, 'heart_rate', 80),
    getattr(obs, 'temperature', 37)
)}

❤️ HR: {getattr(obs, 'heart_rate', 80):.0f} bpm
🩺 BP: {getattr(obs, 'blood_pressure', 120):.0f} mmHg
🫁 O2: {getattr(obs, 'oxygen_saturation', 98):.0f}%
🌡️ Temp: {getattr(obs, 'temperature', 37):.1f}°C
"""

    notes = f"""
**Clinical Notes:**
{obs.clinical_notes or 'No notes yet'}

**Warning:** {obs.warning or 'None'}
**Schema Alert:** {getattr(obs, 'schema_alert', 'None')}
"""

    specialist = f"""
**Specialist ({getattr(obs, 'specialist_specialty', 'General')}):**
{getattr(obs, 'specialist_recommendation', 'Not consulted yet')}

Confidence: {getattr(obs, 'specialist_confidence', 0):.0%}
Reasoning: {getattr(obs, 'specialist_reasoning', 'N/A')}
"""

    test_results = "**Test Results:**\n"
    if obs.test_results:
        for test, result in obs.test_results.items():
            test_results += f"✅ **{test}**: {result}\n"
    else:
        test_results += "No tests ordered yet."

    reward_text = f"""
**Total Reward: {obs.reward:.3f}**

Protocol:     {getattr(obs, 'reward_protocol', 0):.3f}
Diagnosis:    {getattr(obs, 'reward_diagnosis', 0):.3f}
Specialist:   {getattr(obs, 'reward_specialist', 0):.3f}
Efficiency:   {getattr(obs, 'reward_efficiency', 0):.3f}
Safety:       {getattr(obs, 'reward_safety', 0):.3f}
Anti-Exploit: {getattr(obs, 'reward_anti_exploit', 0):.3f}
Token Eff:    {getattr(obs, 'reward_token_efficiency', 0):.3f}
Schema Adapt: {getattr(obs, 'reward_schema_adaptation', 0):.3f}
"""

    status = "✅ EPISODE COMPLETE - Click Reset" if getattr(obs, 'done', False) else "🔄 IN PROGRESS"

    return (
        patient_info,
        vitals,
        notes,
        specialist,
        test_results,
        reward_text,
        status
    )


def reset_ui(difficulty="easy"):
    global current_obs, env, current_difficulty
    current_difficulty = difficulty
    env = PulseEnvironment()
    current_obs = env.reset(difficulty=difficulty)
    return update_ui(current_obs)


def take_action_ui(action_type, content, reasoning):
    global current_obs
    if current_obs is None or getattr(current_obs, 'done', False):
        return update_ui(None)

    action = PulseAction(
        action_type=action_type,
        content=content.strip(),
        reasoning=reasoning.strip()
    )

    current_obs = env.step(action)
    return update_ui(current_obs)


# ====================== GRADIO UI ======================
with gr.Blocks(
    title="🫀 PULSE - Clinical Protocol AI"
) as demo:

    gr.HTML("""
    <div class="pulse-header">
        <h1>🫀 PULSE</h1>
        <h2>Protocol Understanding for Life-Saving Evidence</h2>
        <p><strong>Team Axiom • IIT Madras • Meta PyTorch OpenEnv Hackathon 2026</strong></p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Difficulty Level",
                info="Easy → Medium → Hard (Auto escalates)"
            )
            reset_button = gr.Button("🔄 NEW PATIENT", variant="primary", size="large")

        with gr.Column(scale=2):
            patient_box = gr.Textbox(
                label="👤 Patient Case",
                lines=6,
                interactive=False,
                container=True
            )

    with gr.Row():
        with gr.Column():
            vitals_box = gr.Textbox(
                label="📊 Vital Signs Monitor",
                lines=8,
                interactive=False
            )
            specialist_box = gr.Textbox(
                label="🩺 Specialist Consultation",
                lines=6,
                interactive=False
            )

        with gr.Column():
            notes_box = gr.Textbox(
                label="📋 Clinical Notes & Alerts",
                lines=8,
                interactive=False
            )
            test_box = gr.Textbox(
                label="🔬 Test Results",
                lines=6,
                interactive=False
            )

    with gr.Row():
        reward_box = gr.Textbox(
            label="⭐ Reward Breakdown (8 Components)",
            lines=10,
            interactive=False
        )
        status_box = gr.Textbox(
            label="🎯 Status",
            value="Click Reset to start",
            interactive=False
        )

    gr.Markdown("### 🎮 Take Action")

    with gr.Row():
        action_dropdown = gr.Dropdown(
            choices=[
                "request_history", "physical_exam", "order_test",
                "consult_specialist", "override_specialist",
                "submit_diagnosis", "wait"
            ],
            value="request_history",
            label="Action Type",
            scale=1
        )
        content_input = gr.Textbox(
            label="Content (test name or diagnosis)",
            placeholder="e.g. urinalysis or Urinary Tract Infection",
            scale=2
        )
        reasoning_input = gr.Textbox(
            label="Reasoning (keep short - Mercor bonus)",
            placeholder="Patient has classic symptoms...",
            scale=2
        )

    action_button = gr.Button("▶ EXECUTE ACTION", variant="primary", size="large")

    # Connect buttons
    reset_button.click(
        reset_ui,
        inputs=[difficulty],
        outputs=[patient_box, vitals_box, notes_box, specialist_box, test_box, reward_box, status_box]
    )

    action_button.click(
        take_action_ui,
        inputs=[action_dropdown, content_input, reasoning_input],
        outputs=[patient_box, vitals_box, notes_box, specialist_box, test_box, reward_box, status_box]
    )

    gr.Markdown("""
    **PULSE** — Built for Meta PyTorch OpenEnv Hackathon 2026  
    Team Axiom • Rauank Ratan, Akash Deep, Sangam Jha (IIT Madras)
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=UI_THEME,
        css=UI_CSS,
    )