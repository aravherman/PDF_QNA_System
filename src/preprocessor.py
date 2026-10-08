import re


def clean_text(text):
    """
    Clean extracted PDF text.
    """

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Remove unnecessary special characters
    text = re.sub(r"[^\w\s.,!?;:()'\"%+-]", "", text)

    return text.strip()