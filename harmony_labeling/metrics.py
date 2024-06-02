import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_bach_metrics(predicted_labels, true_labels, evaluation_type="full_chord"):
    """
    Calculates Event-level accuracy (AccE), Segment-level precision (PS),
    recall (RS), and F-measure (FS) for BaCh dataset.

    Args:
        predicted_labels (list): List of predicted chord labels for each frame.
        true_labels (list): List of ground truth chord labels for each frame.
        evaluation_type (str): "full_chord" or "root_only"

    Returns:
        dict: Dictionary containing AccE, PS, RS, and FS scores.
    """

    if evaluation_type not in ["full_chord", "root_only"]:
        raise ValueError("Invalid evaluation_type. Choose from 'full_chord' or 'root_only'")

    # Event-level accuracy (AccE)
    acc_e = calculate_event_accuracy(predicted_labels, true_labels) * 100

    # Segment-level metrics
    predicted_segments = extract_segments(predicted_labels)
    true_segments = extract_segments(true_labels)

    ps = calculate_segment_precision(predicted_segments, true_segments) * 100
    rs = calculate_segment_recall(predicted_segments, true_segments) * 100
    fs = calculate_segment_f_measure(ps, rs) * 100

    return {
        'AccE': acc_e,
        'PS': ps,
        'RS': rs,
        'FS': fs,
    }


def extract_segments(labels):
    """Extracts segments from a list of chord labels."""
    segments = []
    current_segment = [0, labels[0]]
    for i in range(1, len(labels)):
        if labels[i] != current_segment[1]:
            current_segment[0] = i
            segments.append(current_segment)
            current_segment = [i, labels[i]]
    segments.append(current_segment)
    return segments


def calculate_event_accuracy(predicted_labels, true_labels):
    correct_events = sum([1 for pred, true in zip(predicted_labels, true_labels) if pred == true])
    total_events = len(true_labels)
    acc_e = (correct_events / total_events)
    return acc_e


def calculate_segment_precision(predicted_segments, true_segments):
    """Calculates segment-level precision."""
    correct_segments = 0
    for p_segment in predicted_segments:
        for t_segment in true_segments:
            if p_segment == t_segment:
                correct_segments += 1
                break
    return correct_segments / len(predicted_segments)


def calculate_segment_recall(predicted_segments, true_segments):
    """Calculates segment-level recall."""
    correct_segments = 0
    for t_segment in true_segments:
        for p_segment in predicted_segments:
            if p_segment == t_segment:
                correct_segments += 1
                break
    return correct_segments / len(true_segments)


def calculate_segment_f_measure(precision, recall):
    """Calculates segment-level F-measure."""
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)