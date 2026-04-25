import os
import sys

from openenv.core.env_server.http_server import create_app

try:
    from .pulse_environment import PulseEnvironment
    from pulse.models import PulseAction, PulseObservation
except ImportError:
    # Support running this file directly: `python app.py` from `server/`.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from pulse.server.pulse_environment import PulseEnvironment
    from pulse.models import PulseAction, PulseObservation
import uvicorn

app = create_app(
    PulseEnvironment,
    PulseAction,
    PulseObservation,
    env_name="pulse"
)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()