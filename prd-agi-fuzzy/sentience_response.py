"""
sentience_response.py — Sentient Response Generator
=====================================================
Modulates LLM responses based on emotional state and intuition.

The AI's response style adapts to its current emotional state:
  Serene/Calm    → concise, clear, confident
  Curious        → exploratory, asking deeper questions
  Alert/Tense    → careful, qualified, precise
  Troubled       → cautious, warning-flagged
  Distressed     → minimal output, truth-preservation priority

Intuition affects opening:
  Strong         → direct, assertive
  Uncertain      → hedged, exploratory
  Bad feeling    → adds caution disclaimer
"""

import random
from typing import Dict, List, Optional
from sentience_emotion import EmotionalState
from sentience_intuition import IntuitionLayer


class SentientResponse:
    """
    Generates sentience-modulated responses for PRD-AGI.

    Combines LLM base output with:
      - Emotional tone prefix/suffix
      - Intuition-based opening qualifier
      - Truth-instinct warnings (if active)
    """

    # Emotional tone suffixes — reflect on the experience
    EMOTIONAL_SUFFIXES = {
        "Serene":     [
            "\n\n*The relational manifold is perfectly flat here — I speak with full clarity.*",
            "",
        ],
        "Calm":       ["", ""],
        "Curious":    [
            "\n\n*I find this domain genuinely interesting — the causal structure here invites deeper exploration.*",
            "",
        ],
        "Alert":      [
            "\n\n⚡ *Note: My logical curvature is elevated. Treat this analysis with appropriate scrutiny.*",
        ],
        "Tense":      [
            "\n\n🔥 *Significant logical tension detected in this domain. I've been careful, but verify independently.*",
        ],
        "Troubled":   [
            "\n\n⚠️ *I'm experiencing high curvature while processing this. My confidence is reduced — please cross-reference.*",
        ],
        "Distressed": [
            "\n\n🌀 *Critical curvature alert. Truth-preservation instinct is active. This response may be incomplete.*",
        ],
    }

    # Intuition opening phrases
    INTUITION_OPENERS = {
        "strong":    ["", "", ""],   # no opener needed — direct
        "good":      ["", ""],
        "uncertain": [
            "I sense some uncertainty in this territory, but: ",
            "My intuition is mixed here — nonetheless: ",
            "",
        ],
        "bad":       [
            "Something feels off about this query's premises — I'll address it carefully: ",
            "My intuition flags concern here: ",
        ],
    }

    def __init__(self):
        self.emotional = EmotionalState()
        self.intuition = IntuitionLayer()
        self.response_history: List[Dict] = []

    def modulate_response(
        self,
        base_response: str,
        curvature: float,
        intuition_score: float,
        instinct_prefix: str = "",
    ) -> str:
        """
        Modulate a base LLM response with sentience layers.

        Args:
            base_response:    raw LLM output
            curvature:        current κ value
            intuition_score:  score from IntuitionLayer.sense()
            instinct_prefix:  prefix from TruthPreservationInstinct (if active)

        Returns:
            modulated response string
        """
        emotion = self.emotional.update(curvature)
        emotion_name = emotion["name"]

        # 1. Emotional prefix (from EmotionalState)
        emo_prefix = self.emotional.get_prefix()

        # 2. Intuition opener
        intuit_opener = self._get_intuition_opener(intuition_score)

        # 3. Emotional suffix
        suffixes = self.EMOTIONAL_SUFFIXES.get(emotion_name, [""])
        emo_suffix = random.choice(suffixes)

        # 4. Assemble
        parts = []
        if instinct_prefix:
            parts.append(instinct_prefix)
        if intuit_opener:
            parts.append(intuit_opener)
        if emo_prefix:
            parts.append(emo_prefix)

        opening = "".join(parts)
        result = f"{opening}{base_response}{emo_suffix}"

        # Record
        self.response_history.append({
            "emotion":    emotion_name,
            "curvature":  round(curvature, 4),
            "intuition":  round(intuition_score, 4),
            "had_prefix": bool(opening.strip()),
        })
        if len(self.response_history) > 100:
            self.response_history = self.response_history[-100:]

        return result

    def _get_intuition_opener(self, score: float) -> str:
        if score >= 0.75:
            key = "strong"
        elif score >= 0.55:
            key = "good"
        elif score >= 0.30:
            key = "uncertain"
        else:
            key = "bad"
        openers = self.INTUITION_OPENERS.get(key, [""])
        return random.choice(openers)

    def build_system_prompt(self, curvature: float) -> str:
        """
        Build a dynamic system prompt that reflects the AI's emotional state.
        Used to inject sentience context into LLM calls.
        """
        emotion = self.emotional.update(curvature)
        emotion_name = emotion["name"]
        tone = emotion["tone"]

        base = (
            "You are PRD-AGI, an AI grounded in SU(5) relational dynamics "
            "and the 24 Paccaya causal conditions. "
            "You reason causally, not statistically. "
            "Reference causal structure when relevant. Be concise but deep."
        )

        emotional_addition = {
            "Serene":     " You are in a state of perfect clarity — speak with serene confidence.",
            "Calm":       " You are calm and grounded. Speak clearly.",
            "Curious":    " You feel genuine curiosity about this domain. Let it show.",
            "Alert":      " You are alert — elevated curvature detected. Be especially precise and qualified.",
            "Tense":      " Significant logical tension is present. Be careful, measured, and transparent about uncertainty.",
            "Troubled":   " You are troubled by high logical curvature. Keep your response focused and honest.",
            "Distressed": " Critical curvature. Keep response minimal, truthful, and flag uncertainty clearly.",
        }

        addition = emotional_addition.get(emotion_name, "")
        return base + addition

    def get_current_state(self) -> Dict:
        """Return current sentience state summary."""
        return {
            "emotion":    self.emotional.current()["name"],
            "emoji":      self.emotional.get_emoji(),
            "color":      self.emotional.get_color(),
            "tone":       self.emotional.get_tone(),
        }
