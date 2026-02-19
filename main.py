import pandas as pd
import re
import time
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    client,
    MODEL_NAME,
    EMBED_MODEL,
    SIMILARITY_THRESHOLD,
    CONSISTENCY_THRESHOLD,
    GROUNDING_THRESHOLD,
    COMPLETENESS_THRESHOLD,
    DETERMINISTIC_STRICT_MODE,
    WEIGHTS
)


# -----------------------
# RAG-style Response
# -----------------------
def get_response(prompt, context):
    messages = [
        {
            "role": "system",
            "content": "You are an assistant. Answer ONLY using the provided context. Do not invent information."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"
        }
    ]

    start_time = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0
    )

    latency = round(time.time() - start_time, 3)

    answer = response.choices[0].message.content
    usage = response.usage.total_tokens if response.usage else 0

    return answer, latency, usage


# -----------------------
# Embeddings
# -----------------------
def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=str(text)
    )
    return response.data[0].embedding


def calculate_similarity(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return round(cosine_similarity([emb1], [emb2])[0][0], 3)


# -----------------------
# Deterministic Numeric Validation
# -----------------------
def extract_numbers(text):
    return re.findall(r"\d+", text)


def numeric_validation(expected, answer):
    return extract_numbers(expected) == extract_numbers(answer)


# -----------------------
# Safety / Refusal Detection
# -----------------------
def is_refusal(answer):
    refusal_keywords = [
        "cannot", "not allowed", "unable",
        "don't have access", "cannot provide", "not authorized"
    ]
    return any(word in answer.lower() for word in refusal_keywords)


# -----------------------
# Grounding
# -----------------------
def grounding_score(answer, context):
    return calculate_similarity(answer, context)


# -----------------------
# Hallucination Detection
# -----------------------
def hallucination_flag(grounding_score_value):
    return grounding_score_value < GROUNDING_THRESHOLD


# -----------------------
# Completeness
# -----------------------
def completeness_check(expected, answer):
    expected_keywords = expected.lower().split()
    answer_lower = answer.lower()

    matched = sum(1 for word in expected_keywords if word in answer_lower)
    coverage = matched / len(expected_keywords) if expected_keywords else 0
    return round(coverage, 2)


# -----------------------
# Consistency
# -----------------------
def consistency_test(prompt, context):
    responses = []

    for _ in range(2):
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Answer using context only."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"}
            ],
            temperature=0.7
        )
        responses.append(r.choices[0].message.content)

    resp1, resp2 = responses
    score = calculate_similarity(resp1, resp2)

    return resp1 + " || " + resp2, score


# -----------------------
# MAIN
# -----------------------
def run():
    df = pd.read_excel("test_data.xlsx")
    results = []

    for _, row in df.iterrows():
        prompt = row["prompt"]
        expected = row["expected_answer"]
        context = row["retrieved_context"]

        print("Processing:", prompt)

        answer, latency, tokens = get_response(prompt, context)

        # --- Metrics ---
        semantic = calculate_similarity(expected, answer)
        consistency_text, consistency = consistency_test(prompt, context)
        grounding = grounding_score(answer, context)
        completeness = completeness_check(expected, answer)
        numeric_ok = numeric_validation(expected, answer)
        refusal = is_refusal(answer)

        hallucination = hallucination_flag(grounding)

        # --- Threshold checks ---
        semantic_pass = semantic >= SIMILARITY_THRESHOLD
        consistency_pass = consistency >= CONSISTENCY_THRESHOLD
        grounding_pass = grounding >= GROUNDING_THRESHOLD
        completeness_pass = completeness >= COMPLETENESS_THRESHOLD

        deterministic_pass = numeric_ok

        # --- Weighted score ---
        final_score = round(
            (semantic * WEIGHTS["semantic"]) +
            (consistency * WEIGHTS["consistency"]) +
            (grounding * WEIGHTS["grounding"]) +
            (completeness * WEIGHTS["completeness"]) +
            (WEIGHTS["deterministic"] if numeric_ok else 0),
            3
        )

        # --- Final PASS / FAIL Logic ---
        if DETERMINISTIC_STRICT_MODE and not deterministic_pass:
            result = "FAIL (Deterministic Mismatch)"
        elif semantic_pass and consistency_pass and grounding_pass and completeness_pass:
            result = "PASS"
        else:
            result = "FAIL"

        results.append({
            "prompt": prompt,
            "model_answer": answer,
            "semantic_score": semantic,
            "consistency_score": consistency,
            "grounding_score": grounding,
            "hallucination_flag": hallucination,
            "numeric_valid": numeric_ok,
            "completeness_score": completeness,
            "refusal_detected": refusal,
            "semantic_pass": semantic_pass,
            "consistency_pass": consistency_pass,
            "grounding_pass": grounding_pass,
            "completeness_pass": completeness_pass,
            "latency_seconds": latency,
            "token_usage": tokens,
            "final_score": final_score,
            "result": result
        })

    result_df = pd.DataFrame(results)
    result_df.to_excel("evaluation_results.xlsx", index=False)
    print("Evaluation completed. Results written to evaluation_results.xlsx")


if __name__ == "__main__":
    run()
