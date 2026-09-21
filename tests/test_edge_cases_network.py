"""Edge-case test suite for UDP networking, speed bounds, and concurrency."""

import threading
import time
import unittest
from src.comm.protocol import RobotCommand
from src.comm.udp_transmitter import UDPTransmitter


class TestEdgeCasesNetwork(unittest.TestCase):
    def test_speed_bounds_clamping(self):
        """Verify PWM speed clamping between 0 and 255."""
        cmd_neg = RobotCommand(action="FORWARD", speed=-50)
        self.assertEqual(cmd_neg.speed, 0)

        cmd_over = RobotCommand(action="FORWARD", speed=999)
        self.assertEqual(cmd_over.speed, 255)

    def test_dynamic_target_switch_while_streaming(self):
        """Transmitter can safely switch target IP/port on the fly while streaming."""
        tx = UDPTransmitter(target_ip="127.0.0.1", target_port=49200, send_rate_hz=25.0)
        tx.start()
        try:
            tx.update_command("FORWARD", 200)
            time.sleep(0.08)
            # Switch target mid-stream
            tx.set_target("127.0.0.1", 49201)
            time.sleep(0.08)
            telemetry = tx.get_telemetry()
            self.assertEqual(telemetry["target_port"], 49201)
            self.assertGreater(telemetry["packets_sent"], 1)
        finally:
            tx.stop()

    def test_concurrent_emergency_stops(self):
        """Verify thread safety when multiple threads fire commands simultaneously."""
        tx = UDPTransmitter(target_ip="127.0.0.1", target_port=49202, send_rate_hz=50.0)
        tx.start()

        def spam_worker():
            for _ in range(25):
                tx.update_command("FORWARD", 200)
                tx.send_emergency_stop()

        threads = [threading.Thread(target=spam_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final state check
        tx.stop()
        self.assertGreater(tx.packets_sent, 0)


if __name__ == "__main__":
    unittest.main()
