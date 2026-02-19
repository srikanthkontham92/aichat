from openai import OpenAI

# 🔑 
API_KEY = 
client = OpenAI(api_key=API_KEY)

MODEL_NAME = "gpt-4.1-nano"
EMBED_MODEL = "text-embedding-3-small"

SIMILARITY_THRESHOLD = 0.80
CONSISTENCY_THRESHOLD = 0.75

import os
from openai import OpenAI


# -----------------------
# Evaluation Thresholds
# (To be aligned with stakeholders)
# -----------------------

SIMILARITY_THRESHOLD = 0.80
CONSISTENCY_THRESHOLD = 0.75
GROUNDING_THRESHOLD = 0.70
COMPLETENESS_THRESHOLD = 0.75

# Deterministic Numeric Validation Policy
# True = Any mismatch causes automatic FAIL
# False = Deterministic contributes to weighted score only

DETERMINISTIC_STRICT_MODE = True


# -----------------------
# Evaluation Weights
# -----------------------

WEIGHTS = {
    "semantic": 0.30,
    "consistency": 0.20,
    "grounding": 0.20,
    "completeness": 0.10,
    "deterministic": 0.20
}


# -----------------------
# Cost Tracking (Optional Estimate)
# -----------------------

COST_PER_1K_TOKENS = 0.0005  # Adjust based on actual pricing
