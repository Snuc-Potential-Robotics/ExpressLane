"""Automated tests for UDP communication protocol and transmitter."""

import socket
import time
import unittest
from src.comm.protocol import RobotCommand
from src.comm.udp_transmitter import UDPTransmitter


class TestUDPComm(unittest.TestCase):
    def test_robot_command_serialization(self):
        """Test compact string and JSON serialization formats."""
        cmd = RobotCommand(action="FORWARD", speed=215, emotion="Happy", confidence=0.92, seq=42)
        compact = cmd.to_compact_bytes().decode("utf-8")
        self.assertEqual(compact, "F:215:42\n")

        json_bytes = cmd.to_json_bytes().decode("utf-8")
        self.assertIn('"cmd": "F"', json_bytes)
        self.assertIn('"spd": 215', json_bytes)
        self.assertIn('"emo": "Happy"', json_bytes)

    def test_udp_transmission_loop(self):
        """Test UDP socket transmission and reception on localhost."""
        test_port = 49152  # Ephemeral test port
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", test_port))
        receiver.settimeout(1.0)

        tx = UDPTransmitter(target_ip="127.0.0.1", target_port=test_port, send_rate_hz=30.0)
        tx.update_command(action="LEFT", speed=180, emotion="Angry", confidence=0.85)
        tx.start()

        try:
            data, addr = receiver.recvfrom(128)
            decoded = data.decode("utf-8").strip()
            self.assertTrue(decoded.startswith("L:180:"))
        finally:
            tx.stop()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
