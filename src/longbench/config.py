from src.data import INPUT_TAG, CONTEXT_TAG


DATASET2PROMPT = {
    "narrativeqa": f"You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {CONTEXT_TAG}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {INPUT_TAG}\n\nAnswer:",
    "qasper": f'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {CONTEXT_TAG}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {INPUT_TAG}\n\nAnswer:',
    "multifieldqa_en": f"Read the following text and answer briefly.\n\n{CONTEXT_TAG}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "hotpotqa": f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "2wikimqa": f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "musique": f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{CONTEXT_TAG}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:",
    "gov_report": f"You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{CONTEXT_TAG}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": f"You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{CONTEXT_TAG}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {INPUT_TAG}\nAnswer:",
    "multi_news": f"You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{CONTEXT_TAG}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
}

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
}

DATASET2METRIC = {
    "narrativeqa": ["qa_f1_score"],
    "qasper": ["qa_f1_score"],
    "multifieldqa_en": ["qa_f1_score"],
    "hotpotqa": ["qa_f1_score"],
    "2wikimqa": ["qa_f1_score"],
    "musique": ["qa_f1_score"],
    "gov_report": ["rouge_score"],
    "qmsum": ["rouge_score"],
    "multi_news": ["rouge_score"],
}

DATASET2TASK = {
    "narrativeqa": "Single-doc QA",
    "qasper": "Single-doc QA",
    "multifieldqa_en": "Single-doc QA",
    "hotpotqa": "Multi-doc QA",
    "2wikimqa": "Multi-doc QA",
    "musique": "Multi-doc QA",
    "gov_report": "Summarization",
    "qmsum": "Summarization",
    "multi_news": "Summarization",
}
