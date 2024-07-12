
import json 

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

    acc = intersection / gt_labeling_duration
    return acc, intersection, gt_labeling_duration



# def calc_wscr(tracks_gt_labeling, tracks_pred_labeling):
#     pass


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

    acc = intersection / gt_labeling_duration

    return acc, intersection, gt_labeling_duration



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



# \__READ_DATA
def calc_wscr(metrics_file: str,  ignore_zero_chord: bool = True):
    
    if ignore_zero_chord:
        metric_func = calc_normalized_intesection_drop_nochord
    else:
        metric_func = calc_normalized_intesection

        
    with open(metrics_file, 'r') as fp:
        trackId_gt_pred_dict = json.load(fp)
    
    
    mean_acc = 0.
    mean_score = 0.
    
    wscr = 0.
    all_during = 0.
    for track_id, art_track in trackId_gt_pred_dict.items():
        gt_labeling = art_track['gt_labeling']
        pred_labeling = art_track['pred_labeling']
        score = art_track['score']
        
        
        acc, intersection, gt_labeling_duration = metric_func(
            gt_labeling=gt_labeling,
            pred_labeling=pred_labeling
        )
        
        print(f'{acc=}, {intersection=}, {gt_labeling_duration=}')
        #
        wscr += acc * intersection # in the end fraq into all_during
        all_during += gt_labeling_duration

        # 
        mean_acc += acc
        mean_score += score
        print(f'{score=} {mean_score=}')
        
         
    wscr /= all_during
    mean_acc /= len(trackId_gt_pred_dict)
    mean_score /= len(trackId_gt_pred_dict)
    
    return wscr, mean_acc, mean_score
        
        
    

    
# def read_predicts(out_file: str):
#     with open(out_file, 'r') as fp:
#         trackId_gt_pred_dict = json.load(fp)     

    
    

# READ_DATA__/