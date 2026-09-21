"""High-speed UDP transmitter for real-time ESP32 robot control."""

import socket
import threading
import time
import logging
from typing import Dict, Any, Optional
from src.comm.protocol import RobotCommand

logger = logging.getLogger(__name__)


class UDPTransmitter:
    def __init__(
        self,
        target_ip: str = "192.168.4.1",  # Default ESP32 AP mode IP
        target_port: int = 4210,
        send_rate_hz: float = 20.0,
    ):
        self.target_ip = target_ip
        self.target_port = target_port
        self.send_interval = 1.0 / send_rate_hz
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

        # Thread safety & state
        self._lock = threading.Lock()
        self._latest_command = RobotCommand(action="STOP", speed=0)
        self._sequence_counter = 0
        self._running = False
        self._sender_thread: Optional[threading.Thread] = None

        # Telemetry statistics
        self.packets_sent = 0
        self.last_sent_time = 0.0
        self.rtt_ms = 0.0

    def start(self) -> None:
        """Start periodic UDP transmission loop."""
        if self._running:
            return
        self._running = True
        self._sender_thread = threading.Thread(target=self._transmit_loop, daemon=True)
        self._sender_thread.start()
        logger.info(f"UDP Transmitter started, streaming to {self.target_ip}:{self.target_port}")

    def stop(self) -> None:
        """Stop transmitter and send safety STOP packets."""
        self._running = False
        self.send_emergency_stop()
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=0.5)
        try:
            self.sock.close()
        except Exception:
            pass
        logger.info("UDP Transmitter stopped.")

    def set_target(self, ip: str, port: int) -> None:
        """Dynamically update ESP32 target IP and port."""
        with self._lock:
            self.target_ip = ip.strip()
            self.target_port = int(port)
        logger.info(f"UDP Target updated: {self.target_ip}:{self.target_port}")

    def update_command(self, action: str, speed: int = 200, emotion: str = "Neutral", confidence: float = 1.0) -> None:
        """Update current command to be streamed to ESP32."""
        with self._lock:
            self._sequence_counter += 1
            self._latest_command = RobotCommand(
                action=action,
                speed=speed,
                emotion=emotion,
                confidence=confidence,
                seq=self._sequence_counter,
                timestamp=time.time(),
            )

    def send_emergency_stop(self) -> None:
        """Immediately emit STOP command 3 times for redundancy."""
        with self._lock:
            self._sequence_counter += 1
            cmd = RobotCommand(action="STOP", speed=0, emotion="Neutral", confidence=1.0, seq=self._sequence_counter)
            packet = cmd.to_compact_bytes()
            for _ in range(3):
                try:
                    self.sock.sendto(packet, (self.target_ip, self.target_port))
                except Exception as exc:
                    logger.debug(f"Failed to send emergency stop: {exc}")

    def _transmit_loop(self) -> None:
        """Continuous transmission loop feeding ESP32 watchdog timer."""
        while self._running:
            start_time = time.perf_counter()

            with self._lock:
                cmd = self._latest_command
                dest = (self.target_ip, self.target_port)

            packet = cmd.to_compact_bytes()
            try:
                self.sock.sendto(packet, dest)
                self.packets_sent += 1
                self.last_sent_time = time.time()
            except Exception as exc:
                logger.debug(f"UDP send error to {dest}: {exc}")

            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.001, self.send_interval - elapsed)
            time.sleep(sleep_time)

    def get_telemetry(self) -> Dict[str, Any]:
        """Return transmitter status and statistics."""
        with self._lock:
            cmd = self._latest_command
            return {
                "target_ip": self.target_ip,
                "target_port": self.target_port,
                "packets_sent": self.packets_sent,
                "active_cmd": cmd.cmd_char,
                "active_action": cmd.action,
                "speed": cmd.speed,
                "seq": cmd.seq,
                "rtt_ms": round(self.rtt_ms, 1),
            }
