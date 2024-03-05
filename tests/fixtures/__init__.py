"""Helpful fixtures for configuring individual unit tests.
"""
# Standard
import os

### Constants used in fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__))
TINY_MODELS_DIR = os.path.join(FIXTURES_DIR, "tiny_models")
CAUSAL_LM_MODEL = os.path.join(TINY_MODELS_DIR, "LlamaForCausalLM")
