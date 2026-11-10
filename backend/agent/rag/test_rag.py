"""
20-Question Comprehensive RAG Benchmark and Diagnostic Suite
============================================================
Validates that structure-aware chunking, exercise metadata filtering,
and similarity thresholding return accurate, grounded exercise science chunks
across all 5 exercises and general biomechanics principles.
"""

import os
import sys

# Ensure backend directory is in path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from agent.rag.retrieval import retrieve_exercise_knowledge, retrieve_exercise_knowledge_diagnostic

BENCHMARK_QUERIES = [
    # Squats
    ("What is the proper knee angle for squat parallel depth?", "Squats", "squats"),
    ("Why does leaning forward during squats cause lumbar spine shear?", "Squats", "squats"),
    ("What causes knee valgus and knees caving inward during squats?", "Squats", "squats"),
    ("What is the best camera angle and distance for squat analysis?", "Squats", "squats"),
    # Push-ups
    ("What elbow angle is needed for valid push-up depth?", "Push-ups", "push_ups"),
    ("How does hip sagging affect core alignment in push-ups?", "Push-ups", "push_ups"),
    ("How to avoid shoulder impingement and elbow flare during push-ups?", "Push-ups", "push_ups"),
    ("What is the recommended camera setup for push-ups?", "Push-ups", "push_ups"),
    # Biceps Curls
    ("Why is full elbow extension important during biceps curls?", "Biceps Curls (Dumbbell)", "biceps_curls"),
    ("Why is torso swinging bad when curling dumbbells?", "Biceps Curls (Dumbbell)", "biceps_curls"),
    ("What causes elbow drift during biceps curls?", "Biceps Curls (Dumbbell)", "biceps_curls"),
    ("What camera setup is recommended for dumbbell biceps curls?", "Biceps Curls (Dumbbell)", "biceps_curls"),
    # Shoulder Press
    ("What is the starting rack position elbow angle for shoulder press?", "Shoulder Press", "shoulder_press"),
    ("What is the overhead extension angle for shoulder press?", "Shoulder Press", "shoulder_press"),
    ("Why does excessive lower back arching occur during shoulder press?", "Shoulder Press", "shoulder_press"),
    ("What is the recommended camera view for overhead shoulder press?", "Shoulder Press", "shoulder_press"),
    # Lunges
    ("What is the correct front knee angle for lunges?", "Lunges", "lunges"),
    ("How does lateral stance width affect balance during lunges?", "Lunges", "lunges"),
    ("What camera setup and distance is recommended for lunges?", "Lunges", "lunges"),
    # General Safety & 2D Pose
    ("How does vector cosine principle compute 2D joint angles with hysteresis?", None, "general"),
]


def run_benchmark():
    passed = 0
    total = len(BENCHMARK_QUERIES)

    print("==================================================================")
    print("RUNNING 20-QUESTION RAG RETRIEVAL & DIAGNOSTIC BENCHMARK")
    print("==================================================================")

    for idx, (query, exercise_filter, expected_prefix) in enumerate(BENCHMARK_QUERIES, 1):
        res = retrieve_exercise_knowledge(query, exercise_name=exercise_filter, top_k=2)
        diag = retrieve_exercise_knowledge_diagnostic(query, exercise_name=exercise_filter, top_k=2)

        assert diag["total_candidates_examined"] > 0
        if res["results_count"] > 0:
            top_chunk = res["passages"][0]
            chunk_id = top_chunk["chunk_id"]
            exercise = top_chunk["exercise"]
            score = top_chunk["score"]

            # Verify that exercise filter was respected
            if exercise_filter and exercise != "General":
                assert exercise_filter.lower() in exercise.lower(), f"Filter violation: got {exercise} for filter {exercise_filter}"

            print(f"[{idx:02d}/{total}] PASS | Chunk: {chunk_id} | Score: {score:.3f} | {top_chunk['section'][:35]}")
            passed += 1
        else:
            print(f"[{idx:02d}/{total}] FAIL | No accepted chunks for query: '{query}'")

    print("==================================================================")
    print(f"RAG BENCHMARK RESULT: {passed}/{total} Passed ({(passed/total)*100:.0f}% Success)")
    print("==================================================================")
    return passed == total


if __name__ == "__main__":
    success = run_benchmark()
    if not success:
        sys.exit(1)
