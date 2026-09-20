# Corridor Predator Configuration
import os

# WebSocket endpoint (Alchemy / Infura), provided via environment variable
WS_URL = os.getenv("WS_URL", "")

# Simulation Settings
SIMULATION_MODE = True  # Always true for simulation
LOCAL_STATE_FILE = "config/local_state.json"

# Optimization Settings
OPTIMIZATION_BOUNDS = (0.1, 10.0)  # Bounds for parameter a
MAX_ITERATIONS = 100

# Logging
LOG_LEVEL = "INFO"
