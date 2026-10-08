import re


def normalize_text(text):
    """
    Normalize text before comparison.
    """

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def exact_match(predicted, expected):
    """
    Calculate Exact Match.
    """

    predicted = normalize_text(predicted)
    expected = normalize_text(expected)

    return int(predicted == expected)


def calculate_f1(predicted, expected):
    """
    Calculate token-level F1 score.
    """

    predicted_tokens = normalize_text(
        predicted
    ).split()

    expected_tokens = normalize_text(
        expected
    ).split()

    if not predicted_tokens or not expected_tokens:
        return 0.0

    common_tokens = set(
        predicted_tokens
    ) & set(
        expected_tokens
    )

    if not common_tokens:
        return 0.0

    precision = (
        len(common_tokens)
        / len(predicted_tokens)
    )

    recall = (
        len(common_tokens)
        / len(expected_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
    ) / (
        precision + recall
    )


def evaluate_answer(predicted, expected):
    """
    Evaluate one predicted answer.
    """

    return {
        "exact_match": exact_match(
            predicted,
            expected
        ),

        "f1_score": calculate_f1(
            predicted,
            expected
        )
    }


def calculate_overall_metrics(results):
    """
    Calculate overall evaluation metrics.
    """

    if not results:
        return {
            "exact_match": 0,
            "f1_score": 0,
            "confidence": 0
        }

    exact_match_score = sum(
        result["exact_match"]
        for result in results
    ) / len(results)

    f1_score = sum(
        result["f1_score"]
        for result in results
    ) / len(results)

    confidence = sum(
        result["confidence"]
        for result in results
    ) / len(results)

    return {
        "exact_match": exact_match_score,
        "f1_score": f1_score,
        "confidence": confidence
    }