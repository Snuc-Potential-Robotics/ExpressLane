"""Temporal smoother and hysteresis controller with ambiguity guards, full-spectrum decay, and safety failsafes."""

from collections import deque
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default action mappings
DEFAULT_MAPPING = {
    "Happy": "FORWARD",
    "Sad": "BACKWARD",
    "Angry": "LEFT",
    "Surprised": "RIGHT",
    "Neutral": "STOP",
    "Fear": "STOP",
    "Disgust": "STOP",
    "Contempt": "STOP",
}


class TemporalSmoother:
    def __init__(
        self,
        window_size: int = 5,
        alpha: float = 0.45,
        activation_threshold: float = 0.50,
        deactivation_threshold: float = 0.32,
        min_consecutive_frames: int = 3,
        ambiguity_margin: float = 0.10,
        max_missed_frames: int = 3,
        switch_cooldown_frames: int = 2,
        hard_release_raw: float = 0.85,
        turn_actions: tuple = ("LEFT", "RIGHT"),
        turn_hard_release_raw: float = 0.60,
        turn_deactivation_threshold: float = 0.40,
        turn_max_missed_frames: int = 1,
    ):
        self.window_size = window_size
        self.alpha = alpha
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.ambiguity_margin = ambiguity_margin
        # Short dropout grace: ride through 1-3 missed frames (~40-120ms @25fps)
        # instead of stuttering STOP on every detector flicker. Failsafe still
        # engages quickly, far inside the 350ms hardware watchdog.
        self.max_missed_frames = max(0, max_missed_frames)
        # Cooldown after an activation/switch: blocks immediate re-switching
        # so two strong emotions can't oscillate the robot back and forth.
        # Releases to Neutral are never blocked (safety first).
        self.switch_cooldown_frames = max(0, switch_cooldown_frames)
        # A single raw frame this Neutral is an unmistakable relaxation and
        # halts instantly. Softer Neutral surges must persist through the EMA
        # first, so one noisy frame can't chop a held command into a stutter.
        self.hard_release_raw = hard_release_raw

        # Turn-specific precision tuning:
        # Turning requires sharp, on-a-dime precision without lingering angular inertia or overshoot.
        self.turn_actions = tuple(turn_actions)
        self.turn_hard_release_raw = turn_hard_release_raw
        self.turn_deactivation_threshold = turn_deactivation_threshold
        self.turn_max_missed_frames = max(0, turn_max_missed_frames)

        # State tracking
        self.smoothed_scores: Dict[str, float] = {}
        self.recent_dominant_history = deque(maxlen=window_size)
        self.active_emotion: str = "Neutral"
        self.active_action: str = "STOP"
        self.consecutive_count: int = 0
        self.candidate_emotion: str = "Neutral"
        self.missed_count: int = 0
        self.switch_cooldown: int = 0
        # Frames we've held a below-threshold active command while a switch
        # candidate is forming (bounded so noise can't pin a stale command).
        self.decayed_hold: int = 0

        # Mapping dict
        self.mapping = dict(DEFAULT_MAPPING)

    def update_mapping(self, new_mapping: Dict[str, str]) -> None:
        """Update emotion to action mapping."""
        self.mapping.update(new_mapping)

    def reset(self) -> None:
        """Force full reset to the Neutral/STOP failsafe state."""
        self.smoothed_scores = {k: (1.0 if k == "Neutral" else 0.0) for k in DEFAULT_MAPPING.keys()}
        self.recent_dominant_history.clear()
        self.active_emotion = "Neutral"
        self.active_action = self.mapping.get("Neutral", "STOP")
        self.consecutive_count = 0
        self.candidate_emotion = "Neutral"
        self.missed_count = 0
        self.switch_cooldown = 0
        self.decayed_hold = 0

    def is_ambiguous(self, sorted_scores: list) -> bool:
        """
        Check if the top scores are too close to call unambiguously.
        E.g., 36% Happy vs 34% Surprised -> Ambiguous -> Safety STOP!
        """
        if len(sorted_scores) < 2:
            return False
        top_name, top_score = sorted_scores[0]
        second_name, second_score = sorted_scores[1]

        # If top score is overwhelmingly high (>= 0.70), it's confident
        if top_score >= 0.70:
            return False

        # If top is non-neutral and runner-up is within ambiguity margin
        if top_name != "Neutral" and (top_score - second_score) < self.ambiguity_margin:
            return True

        return False

    def _adaptive_alpha(self, raw_scores: Dict[str, float]) -> float:
        """Trust history more when the current frame is uncertain.

        Flat/low-confidence frames (peak < 0.45) and near-tie frames get a
        reduced update weight so one noisy inference can't yank the EMA.
        Confident frames update at full speed for responsiveness.
        """
        if not raw_scores:
            return self.alpha
        ordered = sorted(raw_scores.values(), reverse=True)
        peak = ordered[0]
        margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        if peak < 0.45:
            return self.alpha * 0.5
        if margin < self.ambiguity_margin:
            return self.alpha * 0.7
        return self.alpha

    def process(
        self,
        raw_scores: Optional[Dict[str, float]],
        face_detected: bool = True
    ) -> Tuple[str, str, float, Dict[str, float]]:
        """
        Process incoming frame scores and return stabilized emotion and robot command.

        Returns:
            active_emotion: Stabilized emotion name (e.g. "Happy")
            active_action: Robot action ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP")
            active_confidence: Confidence of active emotion
            smoothed_scores: Current dictionary of smoothed emotion probabilities
        """
        # Check whether active action is currently a turn
        current_action = self.mapping.get(self.active_emotion, "STOP")
        is_turn = current_action in self.turn_actions

        # Face missing: grace period holds the last command through brief
        # detector dropouts, then decays to the Neutral/STOP failsafe.
        # For turns, grace is tighter (turn_max_missed_frames) so the robot
        # never spins blindly during tracking dropouts.
        if not face_detected or not raw_scores:
            self.missed_count += 1
            eff_max_missed = self.turn_max_missed_frames if is_turn else self.max_missed_frames
            if self.missed_count <= eff_max_missed:
                # Gentle decay toward Neutral while holding the command.
                for emotion in list(self.smoothed_scores.keys()):
                    if emotion == "Neutral":
                        self.smoothed_scores[emotion] = min(
                            1.0, self.smoothed_scores[emotion] + 0.10
                        )
                    else:
                        self.smoothed_scores[emotion] *= 0.85
                active_confidence = self.smoothed_scores.get(self.active_emotion, 0.0)
                return (
                    self.active_emotion,
                    self.active_action,
                    active_confidence,
                    dict(self.smoothed_scores),
                )
            self.reset()
            return self.active_emotion, self.active_action, 1.0, dict(self.smoothed_scores)

        self.missed_count = 0
        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1

        # 1. Adaptive Exponential Moving Average (EMA) for each emotion score.
        # Decay any emotion not present in raw_scores to avoid stale inertia.
        alpha_eff = self._adaptive_alpha(raw_scores)
        all_emotions = set(self.mapping.keys()) | set(raw_scores.keys())
        for emotion in all_emotions:
            raw_val = raw_scores.get(emotion, 0.0)
            prev_val = self.smoothed_scores.get(emotion, raw_val)
            self.smoothed_scores[emotion] = alpha_eff * raw_val + (1.0 - alpha_eff) * prev_val

        # 2. RAPID RELEASE SAFETY:
        # Halt immediately when the user genuinely relaxes (Neutral surges or
        # the active expression fades with no strong alternative). But when a
        # *different* directional emotion is already strong, this is a
        # deliberate switch, not a relaxation: hold the current command
        # through the debounce instead of dipping to STOP (no stutter).
        # For turns (LEFT/RIGHT), thresholds are sharpened to halt on a dime
        # without lingering inertia or overshoot.
        if self.active_emotion != "Neutral":
            active_raw = raw_scores.get(self.active_emotion, 0.0)
            neutral_raw = raw_scores.get("Neutral", 0.0)
            strong_alternative = any(
                emo != self.active_emotion
                and emo != "Neutral"
                and score >= self.activation_threshold
                for emo, score in raw_scores.items()
            )
            neutral_smoothed = self.smoothed_scores.get("Neutral", 0.0)
            eff_hard_release = self.turn_hard_release_raw if is_turn else self.hard_release_raw
            eff_deact_thresh = self.turn_deactivation_threshold if is_turn else self.deactivation_threshold
            eff_neutral_thresh = min(self.activation_threshold, 0.40) if is_turn else self.activation_threshold

            if (
                neutral_raw >= eff_hard_release
                or neutral_smoothed >= eff_neutral_thresh
                or (active_raw < eff_deact_thresh and not strong_alternative)
            ):
                self.active_emotion = "Neutral"
                self.candidate_emotion = "Neutral"
                self.consecutive_count = 0
                self.switch_cooldown = 0
                self.decayed_hold = 0

        # 3. Identify top dominant smoothed emotions
        sorted_scores = sorted(self.smoothed_scores.items(), key=lambda x: x[1], reverse=True)
        cand_name, cand_score = sorted_scores[0]
        self.recent_dominant_history.append(cand_name)

        # 4. Ambiguity Guard: if the top 2 emotions are nearly tied this frame
        #    is untrustworthy, so it may not activate or switch anything.
        #    It must NOT rewrite the candidate to "Neutral" though: doing that
        #    reset the debounce counter, so a candidate that was merely
        #    borderline on alternate frames could never accumulate the
        #    consecutive frames it needed and the bot would sit still.
        ambiguous = self.is_ambiguous(sorted_scores)

        # 5. Debounce consecutive frame counter (tracks the true candidate)
        if cand_name == self.candidate_emotion:
            self.consecutive_count += 1
        else:
            self.candidate_emotion = cand_name
            self.consecutive_count = 1

        if ambiguous:
            # Hold the count, but block this frame from driving a decision.
            cand_name = "Neutral"
            cand_score = self.smoothed_scores.get("Neutral", 0.5)

        # 6. Hysteresis switching logic (switch cooldown prevents oscillation;
        #    releases to Neutral above are never blocked)
        if self.active_emotion == "Neutral":
            # Need to cross activation threshold and hold for consecutive frames to start moving
            if (
                cand_name != "Neutral"
                and cand_score >= self.activation_threshold
                and self.consecutive_count >= self.min_consecutive_frames
            ):
                self.active_emotion = cand_name
                self.switch_cooldown = self.switch_cooldown_frames
        else:
            # Active movement state. A ready switch takes precedence; a
            # decayed active command is held (bounded) while a concrete
            # alternative is forming so direction changes crossfade instead
            # of dipping through STOP. Anything else releases to Neutral.
            switch_ready = (
                cand_name != self.active_emotion
                and cand_score >= self.activation_threshold
                and self.consecutive_count >= self.min_consecutive_frames
                and self.switch_cooldown == 0
            )
            current_active_score = self.smoothed_scores.get(self.active_emotion, 0.0)
            eff_deact_thresh = self.turn_deactivation_threshold if is_turn else self.deactivation_threshold
            if switch_ready:
                # Smooth switch to another directional command (e.g. Forward -> Left)
                self.active_emotion = cand_name
                self.switch_cooldown = self.switch_cooldown_frames
                self.decayed_hold = 0
            elif current_active_score < eff_deact_thresh:
                forming = (
                    cand_name != "Neutral"
                    and cand_name != self.active_emotion
                    and cand_score >= self.deactivation_threshold
                )
                if forming and not is_turn and self.decayed_hold < self.min_consecutive_frames:
                    self.decayed_hold += 1
                else:
                    self.active_emotion = "Neutral"
                    self.consecutive_count = 0
                    self.switch_cooldown = 0
                    self.decayed_hold = 0
            else:
                self.decayed_hold = 0

        self.active_action = self.mapping.get(self.active_emotion, "STOP")
        active_confidence = self.smoothed_scores.get(self.active_emotion, 0.0)

        return self.active_emotion, self.active_action, active_confidence, dict(self.smoothed_scores)
