def detect_and_penalize_repetition(current_text: str, previous_text: str, threshold: float = 0.8) -> bool:
    """
    Detects if the current text is highly repetitive compared to the previous text.
    If it repeats a large prefix, it returns True (indicating repetition).
    """
    if not previous_text:
        return False

    # Check if a large portion of previous text is a prefix of current text
    prefix_length = int(len(previous_text) * threshold)
    if prefix_length > 0 and current_text.startswith(previous_text[:prefix_length]):
        return True

    return False
