import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter


def calculate_bach_metrics(predicted_labels, true_labels, evaluation_type="full_chord", 
                           union_segments: bool = False):
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
    predicted_segments = extract_segments(predicted_labels, union_segments=union_segments)
    true_segments = extract_segments(true_labels, union_segments=union_segments)

    # print(predicted_segments)
    # print(true_segments)
    # ps = calculate_segment_precision(predicted_segments, true_segments) * 100
    # rs = calculate_segment_recall(predicted_segments, true_segments) * 100
    # fs = calculate_segment_f_measure(ps / 100, rs / 100) * 100

    metrics_dict = weighted_segment_metrics_per_label(
        predicted_segments=predicted_segments,
        true_segments=true_segments
    )
    
    metrics_dict = metrics_dict['micro_metrics']
    ps = metrics_dict['precision']
    rs =  metrics_dict['recall']
    fs = metrics_dict['f1']
    return {
        'AccE': acc_e,
        'PS': ps * 100,
        'RS': rs * 100,
        'FS': fs * 100,
    }


def extract_segments(labels, union_segments: bool = False):
    """Extracts segments from a list of chord labels."""
    if not union_segments:
        return [(i, label) for i, label in enumerate(labels)]
    
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


# def calculate_segment_precision(predicted_segments, true_segments):
#     """Calculates segment-level precision."""
#     correct_segments = 0
#     for p_segment in predicted_segments:
#         for t_segment in true_segments:
#             if p_segment == t_segment:
#                 correct_segments += 1
#                 break
#     return correct_segments / len(predicted_segments)


# def calculate_segment_recall(predicted_segments, true_segments):
#     """Calculates segment-level recall."""
#     correct_segments = 0
#     for t_segment in true_segments:
#         for p_segment in predicted_segments:
#             if p_segment == t_segment:
#                 correct_segments += 1
#                 break
#     return correct_segments / len(true_segments)


# def calculate_segment_f_measure(precision, recall):
#     """Calculates segment-level F-measure."""
#     if precision + recall == 0:
#         return 0
#     return (2 * precision * recall) / (precision + recall)





def calc_cls_metrics_by_table(true_positives, false_positives, true_negatives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
      'precision': precision,
      'recall': recall,
      'f1': f1,
  }

def metrics_segments(predicted_segments, true_segments):
    """
    Вычисляет  precision, recall и F1 для бинарной классификации сегментов.
    """

    # Создаем множества из кортежей (индекс, лейбл)

    true_positives = 0
    false_positives = 0

    true_negatives = 0
    false_negatives = 0
    for pred_ind, pred_cls in predicted_segments:
        for true_ind, true_cls in true_segments:
            if pred_ind != true_ind:
                continue
            
            if pred_cls == 1:
                if true_cls == 1:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if true_cls == 1:
                    false_negatives += 1
                else:
                    true_negatives += 1
            break

    metrics_dict = calc_cls_metrics_by_table(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives    
    )

    return {
        'metrics': metrics_dict,
        'table': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    }




def weighted_segment_metrics_per_label(predicted_segments, true_segments):
    """
    Вычисляет взвешенные метрики precision, recall и F1 для каждого лейбла сегмента.

    Args:
    predicted_segments: Список кортежей [(индекс, предсказанный_лейбл), ...].
    true_segments: Список кортежей [(индекс, истинный_лейбл), ...].

    Returns:
    Словарь с результатами: {'precision': ..., 'recall': ..., 'f1': ...}.
    """

    label_counts = Counter([label for _, label in true_segments])
    total_segments = len(true_segments)
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    w_true_positives = 0
    w_false_positives = 0
    w_true_negatives = 0
    w_false_negatives = 0
    
    

    for target_label in label_counts:
        # Бинаризация 
        predicted_binary = [(idx, 1 if label == target_label else 0) for idx, label in predicted_segments]
        true_binary = [(idx, 1 if label == target_label else 0) for idx, label in true_segments]

        # Метрики для бинарной задачи
        dct = metrics_segments(predicted_binary, true_binary)
        metrics = dct['metrics']
        
        
        table = dct['table']
        # print(table)
        # print(f"{target_label=}\n{metrics=}")
        
        w_true_positives += table['true_positives']
        w_false_positives += table['false_positives']
        w_true_negatives +=  table['true_negatives']
        w_false_negatives +=  table['false_negatives']

        

        # Взвешивание метрик
        weight = label_counts[target_label] / total_segments
        weighted_precision += metrics['precision'] * weight
        weighted_recall += metrics['recall'] * weight
        weighted_f1 += metrics['f1'] * weight

    
    w_metrics_dict = calc_cls_metrics_by_table(
        true_positives=w_true_positives,
        false_positives=w_false_positives,
        true_negatives=w_true_negatives,
        false_negatives=w_false_negatives    
    )
    
    return {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1,
        
        'micro_metrics': w_metrics_dict
    }


# def calc_WCSR


def calc_CSR(predicted_root_mode_list, gt_root_mode_list):
    track_len = len(predicted_root_mode_list)
    match_cnt = 0
    
    for a_label, b_label in zip(predicted_root_mode_list, gt_root_mode_list):
        if a_label[1] == b_label[1] and a_label[2] == b_label[2]:
            match_cnt += 1
    
    acc = match_cnt / track_len
    return acc, match_cnt, track_len


def calculate_WSCR(
        predicted_trackId_root_mode_list: list[tuple[int, int, int]],
        gt_trackId_root_mode_list: list[tuple[int, int, int]]
    ):
    
    # print(f'{predicted_trackId_root_mode_list=}')
    # print(f'{gt_trackId_root_mode_list=}')

    gt_len = len(gt_trackId_root_mode_list)
    # print(f'{pred_len=} {gt_len=}')
    
    sum_len = len(gt_trackId_root_mode_list)
    
    wscr = 0.0
    
    l = 0
    for r in range(gt_len):
        if gt_trackId_root_mode_list[r][0] != gt_trackId_root_mode_list[l][0] or r == gt_len - 1:
            acc, match_cnt, track_len = calc_CSR(
                predicted_trackId_root_mode_list[l:r],
                gt_trackId_root_mode_list[l:r]
            )
            
            print(f"{acc=}, {match_cnt=}, {track_len=}")
            
            wscr += acc * (track_len / sum_len)
            
            l = r
            
    return wscr

            
    


