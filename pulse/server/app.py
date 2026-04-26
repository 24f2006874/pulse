import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from .ui import demo as gradio_demo
except ImportError:
    try:
        from ui import demo as gradio_demo
    except ImportError:
        gradio_demo = None

from openenv.core.env_server.http_server import create_app
from fastapi.responses import RedirectResponse

try:
    from .pulse_environment import PulseEnvironment
    from pulse.models import PulseAction, PulseObservation
except ImportError:
    # Support running this file directly: `python app.py` from `server/`.
    from pulse.server.pulse_environment import PulseEnvironment
    from pulse.models import PulseAction, PulseObservation
import uvicorn

app = create_app(
    PulseEnvironment,
    PulseAction,
    PulseObservation,
    env_name="pulse"
)


@app.get("/")
def root():
    return {
        "service": "pulse",
        "status": "ok",
        "health": "/health",
        "ui": "/ui",
    }


@app.get("/web")
@app.get("/web/")
@app.get("/webs")
@app.get("/webs/")
def web_alias():
    return RedirectResponse(url="/ui")


if gradio_demo is not None:
    try:
        import gradio as gr

        app = gr.mount_gradio_app(app, gradio_demo, path="/ui")
    except Exception:
        # Keep API server running even if UI mounting fails.
        pass

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()