"""Perception module for PRD-AGI: Ollama + Google Gemini LLM integration."""

from .ollama_client import OllamaInterface
from .gemini_client import GeminiInterface

__all__ = ['OllamaInterface', 'GeminiInterface']
