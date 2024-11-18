from typing import Dict, List
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class Metric:
    @classmethod
    def compute(
        cls,
        predictions: List[str],
        answers: List[List[str]],
        metric_list: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        metric_list = [metric.lower() for metric in metric_list]
        cls._check_metric_list(metric_list)

        result = {}
        for metric in metric_list:
            total_score = 0
            for idx, (prediction, ground_truths) in enumerate(
                zip(predictions, answers)
            ):
                score = 0
                for ground_truth in ground_truths:
                    score = max(
                        score,
                        getattr(cls, metric)(
                            prediction,
                            ground_truth,
                        ),
                    )
                total_score += score
            result[metric] = total_score / len(predictions)

        return result

    @staticmethod
    def _check_metric_list(metric_list: List[str]):
        for metric in metric_list:
            assert hasattr(Metric, metric), f"Not find metric `{metric}`."

    @staticmethod
    def em_score(prediction: str, ground_truth: str) -> float:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction == normalized_ground_truth:
            return 1.0
        else:
            return 0.0
