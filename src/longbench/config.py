from src.data import INPUT_TAG, CONTEXT_TAG


DATASET2PROMPT = {
    "hotpotqa": f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "2wikimqa": f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "musique": f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
}

DATASET2MAXLEN = {
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
}

DATASET2METRIC = {
    "hotpotqa": ["qa_f1_score"],
    "2wikimqa": ["qa_f1_score"],
    "musique": ["qa_f1_score"],
}