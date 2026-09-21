"""Robot communication protocol definitions and serialization."""

import json
import time
from dataclasses import dataclass
from typing import Dict, Any


# Action to command character mapping
ACTION_TO_CMD = {
    "FORWARD": "F",
    "BACKWARD": "B",
    "LEFT": "L",
    "RIGHT": "R",
    "STOP": "S",
}

CMD_TO_ACTION = {v: k for k, v in ACTION_TO_CMD.items()}


@dataclass
class RobotCommand:
    action: str  # FORWARD, BACKWARD, LEFT, RIGHT, STOP
    speed: int = 200  # PWM value (0 - 255)
    emotion: str = "Neutral"
    confidence: float = 1.0
    seq: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        # Clamp speed between 0 and 255
        self.speed = max(0, min(255, int(self.speed)))

    @property
    def cmd_char(self) -> str:
        return ACTION_TO_CMD.get(self.action.upper(), "S")

    def to_compact_bytes(self) -> bytes:
        """
        Ultra-compact format for maximum micro-controller efficiency:
        Format: "CMD:SPEED:SEQ\n", e.g. "F:200:1042\n"
        """
        return f"{self.cmd_char}:{self.speed}:{self.seq}\n".encode("utf-8")

    def to_json_bytes(self) -> bytes:
        """JSON format for rich debugging and telemetry."""
        payload: Dict[str, Any] = {
            "cmd": self.cmd_char,
            "action": self.action,
            "spd": self.speed,
            "emo": self.emotion,
            "conf": round(self.confidence, 2),
            "seq": self.seq,
            "ts": round(self.timestamp, 3),
        }
        return json.dumps(payload).encode("utf-8")
