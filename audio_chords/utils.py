

# \__METRICS

def calc_normalized_intesection_drop_nochord(gt_labeling, pred_labeling):
    intersection = 0

    gt_labeling_ptr = 0
    pred_labeling_ptr = 0

    gt_labeling_duration = 0
    for start, end, label in gt_labeling:
        label = int(label)
        if label == 0:
            continue
        
        gt_labeling_duration += end - start

    while pred_labeling_ptr < len(pred_labeling):
        if gt_labeling_ptr == len(gt_labeling):
            break
    
        # if int(pred_labeling[pred_labeling_ptr][2]) == 0:
        #     # print(f"WTF?!?!??!? if int(pred_labeling[pred_labeling_ptr][2]) == 0:")
        #     # raise Exception
        #     pass
        # else:
        
        if  gt_labeling[gt_labeling_ptr][2] != 0 and gt_labeling[gt_labeling_ptr][2] == pred_labeling[pred_labeling_ptr][2]:
            intersection += max(
                0,
                min(gt_labeling[gt_labeling_ptr][1], pred_labeling[pred_labeling_ptr][1]) - \
                max(gt_labeling[gt_labeling_ptr][0], pred_labeling[pred_labeling_ptr][0])
            )

        if gt_labeling[gt_labeling_ptr][1] >= pred_labeling[pred_labeling_ptr][1]:
            pred_labeling_ptr += 1
        else:
            gt_labeling_ptr += 1

    return intersection / gt_labeling_duration


def calc_normalized_intesection(gt_labeling, pred_labeling):
    intersection = 0

    gt_labeling_ptr = 0
    pred_labeling_ptr = 0

    gt_labeling_duration = 0
    for start, end, _ in gt_labeling:
        gt_labeling_duration += end - start

    while pred_labeling_ptr < len(pred_labeling):
        if gt_labeling[gt_labeling_ptr][2] == pred_labeling[pred_labeling_ptr][2]:
            intersection += max(
                0,
                min(gt_labeling[gt_labeling_ptr][1], pred_labeling[pred_labeling_ptr][1]) - \
                max(gt_labeling[gt_labeling_ptr][0], pred_labeling[pred_labeling_ptr][0])
            )

        if gt_labeling[gt_labeling_ptr][1] >= pred_labeling[pred_labeling_ptr][1]:
            pred_labeling_ptr += 1
        else:
            gt_labeling_ptr += 1

    return intersection / gt_labeling_duration



# METRICS__/

import numpy as np

def calc_sample_weight(targets: np.ndarray):
    positive_ex_count = sum(targets)
    weight = (len(targets) - positive_ex_count) / positive_ex_count
    print(weight)
    return np.where(targets == 1, weight, 1)




import logging

def create_logger(saved_path):
    py_logger = logging.getLogger(__name__)
    py_logger.setLevel(logging.INFO)
    py_handler = logging.FileHandler(saved_path, mode='w')
    py_formatter = logging.Formatter("%(asctime)s %(message)s")

    py_handler.setFormatter(py_formatter)
    py_logger.addHandler(py_handler)
    return py_logger

import os 
os.umask(0)

HALF_SEP = '=============='
SEP = HALF_SEP * 2

DATA_DIR = '/storage/naumtsevalex/harmony/data'
# NPY_DIR = os.path.join(DATA_DIR, 'castom_npy')
OUT_DIR = os.path.join(DATA_DIR, 'results_')
os.makedirs(OUT_DIR, exist_ok=True)
