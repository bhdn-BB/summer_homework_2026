BIO_LABELS = ["O", "B-Mountain", "I-Mountain"]


def token_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    """Return character offsets for whitespace-separated tokens."""
    offsets = []
    cursor = 0

    for token in tokens:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token) + 1

    return offsets


def overlapping_token_indexes(tokens: list[str], start_char: int, end_char: int) -> list[int]:
    """Find token indexes overlapping a character-level entity span."""
    return [
        index
        for index, (token_start, token_end) in enumerate(token_offsets(tokens))
        if token_end > start_char and token_start < end_char
    ]


def spans_to_bio_labels(tokens: list[str], spans: list[dict]) -> list[int]:
    """Convert character-level entity spans into integer BIO labels."""
    labels = [0] * len(tokens)

    for span in spans:
        indexes = overlapping_token_indexes(
            tokens=tokens,
            start_char=span["start"],
            end_char=span["end"],
        )

        if not indexes:
            continue

        labels[indexes[0]] = 1
        for index in indexes[1:]:
            labels[index] = 2

    return labels


def word_spans_to_bio_labels(tokens: list[str], spans: list[dict]) -> list[int]:
    """Convert exclusive word spans into integer BIO labels."""
    labels = [0] * len(tokens)

    for span in spans:
        start_index = max(0, span["word_start_index"])
        end_index = min(len(tokens), span["word_end_index"])
        indexes = list(range(start_index, end_index))

        if not indexes:
            continue

        labels[indexes[0]] = 1
        for index in indexes[1:]:
            labels[index] = 2

    return labels


def seqeval_metrics(true_labels: list[list[int]], pred_labels: list[list[int]]) -> dict:
    """Compute token accuracy and exact-span BIO precision, recall, and F1."""
    def entities(labels: list[int]) -> set[tuple[int, int]]:
        spans = set()
        start = None
        for index, label in enumerate([*labels, 0]):
            if label == 1 or (label == 2 and start is None):
                if start is not None:
                    spans.add((start, index))
                start = index
            elif label != 2 and start is not None:
                spans.add((start, index))
                start = None
        return spans

    true_spans = set()
    predicted_spans = set()
    correct_tokens = 0
    total_tokens = 0
    for row_index, (true_row, predicted_row) in enumerate(zip(true_labels, pred_labels)):
        true_spans.update((row_index, *span) for span in entities(true_row))
        predicted_spans.update((row_index, *span) for span in entities(predicted_row))
        correct_tokens += sum(true == predicted for true, predicted in zip(true_row, predicted_row))
        total_tokens += len(true_row)

    true_positive = len(true_spans & predicted_spans)
    precision = true_positive / len(predicted_spans) if predicted_spans else 0.0
    recall = true_positive / len(true_spans) if true_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "overall_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
    }
