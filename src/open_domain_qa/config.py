from src.data import INPUT_TAG, CONTEXT_TAG


DATASET2PROMPT = {
    "nq": f"Knowledge:\n\n{CONTEXT_TAG}\n\nAnswer the question based on the given knowledge. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "popqa": f"Knowledge:\n\n{CONTEXT_TAG}\n\nAnswer the question based on the given knowledge. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "trivia": f"Knowledge:\n\n{CONTEXT_TAG}\n\nAnswer the question based on the given knowledge. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
}

DATASET2MAXLEN = {
    "nq": 128,
    "popqa": 128,
    "trivia": 128,
}

DATASET2METRIC = {
    "nq": ["em_score"],
    "popqa": ["em_score"],
    "trivia": ["em_score"],
}