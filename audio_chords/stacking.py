import os

import numpy as np
from tqdm import tqdm
from functools import partial
from catboost import CatBoostClassifier
import json

from preprocess_data import load_np, save_np

from train import train, get_accuracy, \
    PARAMS_ROOT, PARAMS_MODE, PARAMS_NOCHORD

SEMITONES = 12

DATA_DIR = '/storage/naumtsevalex/harmony/src/data/root_McGill-Billboard/glinka_chord'
NPY_DIR  = os.path.join(DATA_DIR, 'npy')
CASTOM_NPY_DIR = os.path.join(DATA_DIR, 'castom_npy')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CASTOM_NPY_DIR, exist_ok=True)
RESULT_JSON = os.path.join(MODEL_DIR, 'result.json')


load_np = partial(load_np, data_dir=CASTOM_NPY_DIR)
save_np = partial(save_np, data_dir=CASTOM_NPY_DIR)

def update_json(file, add_json):
    try:
        fp = open(file, mode='r+')
        js = json.load(fp)
    except:
        js = {}
    js.update(add_json)
    with open(file, mode='w') as fp:
        js = json.dump(js, fp, indent=4)

    
# \__PREPROCESS_DATA

# def transform_into_dataset_of_root_probs(model_roots, X_shifts, verbose: bool = True):
#     '''
#     X_shifts = np.zeros((X.shape[0] * SEMITONES, X.shape[1]))
#     # y_shifts = np.zeros((y.shape[0] * SEMITONES))
    
#     '''
#     X_out = []
#     y_out = []
    
#     # (X.shape[0] * SEMITONES, 1)
#     shift_preds_root = model_roots.predict_proba(X_shifts, verbose=verbose)[:, 1]
    
#     # (X.shape[0], SEMITONES)
#     root_probs = np.reshape(shift_preds_root, (X_shifts.shape, SEMITONES))
#     # root_y =  y_shifts[np.arange(0 , -1, SEMITONES)]
    
#     return root_probs


def add_probs_to_dataset(
        model_roots,
        X_shifts, y_shifts, verbose: bool = True):
    '''
    Add prob pred from model_roots 
    Input:
    X_shifts = np.zeros((X_shifts.shape[0], X.shape[1]))
    y_shifts = np.zeros((y_shifts.shape[0]))
    
    Output:
    out_X_shifts = np.zeros((X_shifts.shape[0], X.shape[1] + 1))
    '''
    
    shifts_pred_probs = model_roots.predict_proba(X_shifts, verbose=verbose)[:, 1].reshape(-1, 1)
    # (X_shifts.shape[0], X.shape[1] + 1)
    X_out = np.concatenate([X_shifts, shifts_pred_probs], axis=1)
    y_out = y_shifts
    
    return X_out, y_out
    

# PREPROCESS_DATA__/

# \__TRAIN


def run_train_0(verbose: bool =True):
    (
        model_roots, model_mode, model_nochord, 
        acc_train, acc_test, 
        
        X_train_merged, y_train_merged, X_test_merged, y_test_merged,

        X_train_shifts_merged, y_train_chord_merged, X_test_shifts_merged, # y_test_chord_merged,

        X_train_modes_merged, y_train_modes_merged, X_test_modes_merged, # y_test_modes_merged,
        
        y_train_nochord_merged, y_test_nochord_merged  
    ) = train()
    if verbose:
        print(f"ITER#0:\n{acc_train=}\n{acc_test=}\n")
    # ITER#0
    # acc_train=0.9061677470045177
    # acc_test=0.8503732809430256

    model_roots.save_model(fname=os.path.join(MODEL_DIR, "model_roots_0.chpt"))
    model_mode.save_model(fname=os.path.join(MODEL_DIR, "model_mode_0.chpt"))
    model_nochord.save_model(fname=os.path.join(MODEL_DIR, "model_nochord_0.chpt"))
    
    
    return (
        acc_train, acc_test,
        model_roots, model_mode, model_nochord,
    )

def run_all(
        dont_use_cache_start_with: int = 0,
        iterations=3, verbose: bool = True):
    '''
    if use_cache_util = 0, calculate all models, and datasets
    
    '''
    X_train_merged = load_np("X_train_merged")
    y_train_merged = load_np("y_train_merged")

    X_test_merged = load_np("X_test_merged")
    y_test_merged = load_np("y_test_merged")

    X_train_shifts_merged = load_np("X_train_shifts_merged")
    y_train_chord_merged = load_np("y_train_chord_merged")

    X_test_shifts_merged = load_np("X_test_shifts_merged")
    # y_test_chord_merged = np.load("y_test_chord_merged")

    X_train_modes_merged = load_np("X_train_modes_merged")
    y_train_modes_merged = load_np("y_train_modes_merged")

    X_test_modes_merged = load_np("X_test_modes_merged")
    # y_test_modes_merged = np.load("y_test_modes_merged")
    
    if dont_use_cache_start_with >= 1:
        model_roots = CatBoostClassifier(**PARAMS_ROOT, logging_level="Verbose")
        model_mode = CatBoostClassifier(**PARAMS_MODE, logging_level="Verbose")
        model_nochord = CatBoostClassifier(**PARAMS_NOCHORD, logging_level="Verbose")

        model_roots.load_model(fname=os.path.join(MODEL_DIR, "model_roots_0.chpt"))
        model_mode.load_model(fname=os.path.join(MODEL_DIR, "model_mode_0.chpt"))
        model_nochord.load_model(fname=os.path.join(MODEL_DIR, "model_nochord_0.chpt"))
    else:
        (
            acc_train, acc_test,
            model_roots, model_mode, model_nochord, 
        ) = run_train_0(verbose=verbose)
        
        update_json(RESULT_JSON, {'train_test_0': {
            'acc_train': acc_train,
            'acc_test': acc_test
            }})
    
    for iteration in range(max(dont_use_cache_start_with, 1), iterations):        
        if iteration == 1:
            prev_name_of_X_train_shifts_merged = f'X_train_shifts_merged'
            prev_name_of_X_test_shifts_merged = f'X_test_shifts_merged'
        else:
            prev_name_of_X_train_shifts_merged = f'X_train_shifts_merged_{iteration - 1}'
            prev_name_of_X_test_shifts_merged = f'X_test_shifts_merged_{iteration - 1}'
        
        prev_X_train_shifts_merged = load_np(prev_name_of_X_train_shifts_merged)
        prev_X_test_shifts_merged = load_np(prev_name_of_X_test_shifts_merged)
        
        prev_model_roots = CatBoostClassifier(**PARAMS_ROOT, logging_level="Verbose")
        prev_model_roots.load_model(os.path.join(MODEL_DIR, f'model_roots_{iteration - 1}.chpt'))
        (
            cur_X_train_shifts_merged,
            _
        ) = add_probs_to_dataset(
            model_roots=prev_model_roots,
            X_shifts=prev_X_train_shifts_merged,
            y_shifts=None
        )
        
        (
            cur_X_test_shifts_merged,
            _
        ) = add_probs_to_dataset(
            model_roots=prev_model_roots,
            X_shifts=prev_X_test_shifts_merged,
            y_shifts=None
        )
        save_np(f'X_train_shifts_merged_{iteration}', cur_X_train_shifts_merged)
        save_np(f'X_test_shifts_merged_{iteration}', cur_X_test_shifts_merged)
        
        
        cur_model_roots = CatBoostClassifier(**PARAMS_ROOT, logging_level="Verbose")
        cur_model_roots.fit(cur_X_train_shifts_merged, y_train_chord_merged)
        cur_model_roots.save_model(fname=os.path.join(MODEL_DIR, f"model_roots_{iteration}.chpt"))
        
        acc_train = get_accuracy(
            cur_model_roots,
            model_mode,
            model_nochord,
            X_train_merged,
            cur_X_train_shifts_merged,
            X_train_modes_merged,
            y_train_merged,
        )
        
        acc_test = get_accuracy(
            cur_model_roots,
            model_mode,
            model_nochord,
            X_test_merged,
            cur_X_test_shifts_merged, # X_test_shifts_merged,
            X_test_modes_merged,
            y_test_merged,
        )
            
        update_json(RESULT_JSON, {f'train_test_{iteration}': {
            'acc_train': acc_train,
            'acc_test': acc_test
            }})
         
# TRAIN__/ 




if __name__ == "__main__":
    run_all(dont_use_cache_start_with=0, iterations=5)
    
    
# {
#     "train_test_0": {
#         "acc_train": 0.9102533883323513,
#         "acc_test": 0.8507662082514735
#     },
#     "train_test_1": {
#         "acc_train": 0.9477509330190532,
#         "acc_test": 0.860589390962672
#     },
#     "train_test_2": {
#         "acc_train": 0.9833038695737576,
#         "acc_test": 0.862082514734774
#     },
#     "train_test_3": {
#         "acc_train": 1.0014142604596346,
#         "acc_test": 0.8590176817288802
#     },
#     "train_test_4": {
#         "acc_train": 1.0135926144175997,
#         "acc_test": 0.8554813359528487
#     }
# }

