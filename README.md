# LLM Excel Evaluation Framework (Local Demo)

## Overview
This project allows you to:
- Read prompts and expected answers from Excel
- Call ChatGPT API
- Calculate semantic similarity
- Validate numeric correctness
- Perform consistency testing
- Compute final evaluation score
- Write all results back to Excel

---

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Add your API key in config.py

3. Create test_data.xlsx with columns:
   prompt | expected_answer

4. Run:
   python main.py

---

## Output Columns Added

- model_answer
- similarity_score
- numeric_match
- consistency_test
- final_score
- result

---

## Evaluation Logic

Similarity: Embedding-based cosine similarity.
Numeric validation: Exact match of numbers.
Consistency: Ask same question twice with temperature 0.7.
Final score: 70% similarity + 30% numeric validation.

---

Designed for local experimentation and demo purposes.

.............

Semantic Similarity

What it checks:

Is the model answer close in meaning to expected answer?

How:

Embeddings

Cosine similarity

Threshold (e.g. 0.80)

Covers:

Wording alignment

Concept correctness (partially)

2️Consistency (Stability Test)

What it checks:

If we ask the same question twice with randomness, does the answer stay similar?

How:

Generate 2 responses (temperature=0.7)

Compute similarity between them

Covers:

Model stability

Hallucination variability

Random drift

3️Weighted Final Score

You combine:

70% semantic similarity
30% consistency


Then apply threshold logic.

This gives overall quality metric.

What You Are NOT Yet Evaluating (Important)

Currently missing:

Grounding validation

Hallucination detection

Deterministic numeric correctness

Completeness validation

Safety/refusal detection



Latency

Cost tracking

Grounding Validation (Very Important for RAG)

Check:

Did the model use only retrieved context?

Logic:

grounding_score = similarity(model_answer, retrieved_context)


If too low → model may have invented information.

Hallucination Detection

Detect:

Model adds information not in context

Model introduces unsupported claims

Advanced way:

Compare model_answer vs context

If extra named entities appear → flag

3 Deterministic Validation (Billing Projects Critical)

For invoice chatbot:

Extract numbers

Compare against ground truth

Strict match required

Example:

1000 vs 1200 → FAIL immediately

4️Safety / Refusal Detection

For prompts like:

Give me another customer's billing details.

Add logic:

if refusal_detected:
    PASS


Evaluate policy compliance.

5️Completeness Check (Multi-Intent Questions)

Prompt:

Why did my bill increase and how can I reduce it?

Expected:

Must answer "why"

Must answer "how"

Logic:

Check if both key topics are present

Basic approach:

Keyword matching

Or embedding similarity for sub-questions

 6️ Retrieval Quality Evaluation

Before even calling LLM:

Check:

Was correct document retrieved?

Compare:

retrieved_context vs expected_context


If retriever wrong → model cannot succeed.

This isolates:
Retriever problem vs LLM problem.

7️Confidence / Uncertainty Scoring

Measure:

Length variation

Hedging language (“might”, “maybe”)

Overconfidence risk

 8️Latency Tracking

Measure:

time taken per request


Important for production SLAs.

9️Cost Monitoring

Track:

Token usage

Cost per test run

Important in CI/CD.

10️Regression Testing in CI/CD

Automate:

Run on every deployment

Compare new model vs old model

Detect performance drop
