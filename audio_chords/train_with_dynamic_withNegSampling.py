import argparse
import json
import os 

import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from tqdm import tqdm
import logging

from preprocess_data import load_np, SEMITONES, merge_chords, transform_into_dataset_of_modes,\
    transform_into_dataset_of_shifts, get_no_chord_positives, create_negative_root_chord
from train import PARAMS_MODE, PARAMS_NOCHORD, PARAMS_ROOT
from dynamics import harmony_seq_pipeline
from utils import calc_normalized_intesection, calc_normalized_intesection_drop_nochord,\
    create_logger, calc_sample_weight, \
    HALF_SEP, SEP, DATA_DIR, OUT_DIR




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_id', type=int, default=0)
    args = parser.parse_args()
    
    # my_logger.info(f'{args.fold_id=}')
    # my_logger.info(f'{type(args.fold_id)=}')

    # X = np.load("/home/azatvaleev/audiochord/01_all_chroma_vectors.npy")
    # y = np.load("/home/azatvaleev/audiochord/01_all_chord_labels_extended.npy")
    X = np.load("/storage/naumtsevalex/harmony/data/castom_npy/01_all_chroma_vectors.npy")
    y = np.load("/storage/naumtsevalex/harmony/data/castom_npy/01_all_chord_labels_extended.npy")
    # print(X.shape, len(y), y[0].shape)
    composition_idx = y[:, 0].astype(int)
    time_shifts = y[:, 2]
    labels = y[:, 3].astype(int)

    
    composition_idx_unique = np.unique(composition_idx)
    # my_logger.info(f"{len(composition_idx_unique)=} {len(X)=}")
    # my_logger.info(f"{len(X) / len(composition_idx_unique)=:.2f}")
    
    # composition_idx_unique = composition_idx_unique[:2]
    n_splits=10
    # n_splits=30
    # n_splits=2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    labeling = dict()
    score_sum = 0

    EXP_TITLE = 'woNochord_withNegSamples'
    # EXP_TITLE = 'woNochord_withNegSamples_randomMask060'
    # EXP_TITLE = 'woNochord_withNegSamples_multLastAlpha060'

    for i, (train_indices, test_indices) in enumerate(kf.split(composition_idx_unique)):
        if i == args.fold_id:
            FOLDER_DIR = f'{i:02}_test_folder_nSplits_{n_splits}'
            RESULT_FOLD_DIR = os.path.join(OUT_DIR, FOLDER_DIR)
            os.makedirs(RESULT_FOLD_DIR, exist_ok=True)
            EXP_DIR = os.path.join(RESULT_FOLD_DIR, EXP_TITLE)
            os.makedirs(EXP_DIR, exist_ok=True)
            LABELING_FILE = os.path.join(EXP_DIR, f'{EXP_TITLE}_labeling.json')
            LOGGING_FILE = os.path.join(EXP_DIR, f'{EXP_TITLE}.log')
            
            print(f'{SEP}\nLogs saving in {LOGGING_FILE}\n{SEP}')
            print(f'{EXP_TITLE=} | {n_splits=} | {i=}')
            
            my_logger = create_logger(LOGGING_FILE)
            my_logger.info('Start program...')
            my_logger.info(f'{args.fold_id=}')
            my_logger.info(f'{type(args.fold_id)=}')  
            my_logger.info(f"{len(composition_idx_unique)=} {len(X)=}")
            my_logger.info(f"{len(X) / len(composition_idx_unique)=:.2f}")
            my_logger.info(f"{i=}| {n_splits=} | {len(train_indices)=} | {len(test_indices)=}")
            # ==========
            
            
            
            train_compositions = composition_idx_unique[train_indices]
            test_compositions = composition_idx_unique[test_indices]

            train_mask = [True if composition_idx_ in train_compositions else False for composition_idx_ in composition_idx]

            X_train = X[train_mask]
            y_train = labels[train_mask]

            my_logger.info('X_train_merged, y_train_merged = merge_chords(X_train, y_train)')
            X_train_merged, y_train_merged = merge_chords(X_train, y_train)
            y_train_nochord_merged = np.where(y_train_merged == 0, 1, 0)


            nochord_model = CatBoostClassifier(**PARAMS_NOCHORD, logging_level="Verbose")

            np.random.seed(42)
            my_logger.info('X_nochord_positives, y_nochord_positives = get_no_chord_positives')
            X_nochord_positives, y_nochord_positives = get_no_chord_positives(X_train, 
                                                                              labels[train_mask], 
                                                                              composition_idx[train_mask], 
                                                                              y[train_mask][:, 1:3], 
                                                                              train_compositions, 
                                                                              len(y_train_nochord_merged) - sum(y_train_nochord_merged))
            
            X_nochord_train = np.concatenate((X_train_merged, X_nochord_positives), axis=0)
            y_nochord_train = np.concatenate((y_train_nochord_merged, y_nochord_positives), axis=0)

            permutation = np.random.permutation(len(X_nochord_train))
            X_nochord_train = X_nochord_train[permutation]
            y_nochord_train = y_nochord_train[permutation]

            my_logger.info('nochord_model.fit')
            nochord_model.fit(X_nochord_train, y_nochord_train, sample_weight=calc_sample_weight(y_nochord_train))  # TODO: w/o any shifts

            
            X_train_modes_merged, y_train_modes_merged = transform_into_dataset_of_modes(X_train_merged, y_train_merged)

            # \__add negative samples
            X_negative_train_merged, y_negative_train_merged = create_negative_root_chord(X_train_merged, y_train_merged)
            my_logger.info(f'{100.0 * len(X_negative_train_merged) / len(X_train_merged):.2f}  | {len(X_negative_train_merged)=} {len(X_train_merged)=}')
            
            X_final_train_shifts_merged = np.concatenate((X_train_merged, X_negative_train_merged), axis=0)
            y_final_nochord_train = np.concatenate((y_train_merged, y_negative_train_merged), axis=0)
            permutation = np.random.permutation(len(X_final_train_shifts_merged))
            X_final_train_shifts_merged = X_final_train_shifts_merged[permutation]
            y_final_nochord_train = y_final_nochord_train[permutation]
            # add negative samples__/
            
            (
                X_train_shifts_merged,
                y_train_chord_merged,
                _,
                _,
            ) = transform_into_dataset_of_shifts(X_final_train_shifts_merged, y_final_nochord_train)
            # ) = transform_into_dataset_of_shifts(X_train_merged, y_train_merged)
            
            
            root_model = CatBoostClassifier(**PARAMS_ROOT, logging_level="Verbose")
            mode_model = CatBoostClassifier(**PARAMS_MODE, logging_level="Verbose")
            my_logger.info('root_model.fit')
            root_model.fit(X_train_shifts_merged, y_train_chord_merged, sample_weight=calc_sample_weight(y_train_chord_merged))
            my_logger.info('mode_model.fit')
            mode_model.fit(X_train_modes_merged, y_train_modes_merged, sample_weight=calc_sample_weight(y_train_modes_merged))

            # nochord_model.fit(X_train_merged, y_train_nochord_merged, sample_weight=calc_sample_weight(y_train_nochord_merged))  # TODO: w/o any shifts

            my_logger.info('for test_composition in tqdm(test_compositions):')
            cnt_tracks_wo_chord = 0
            for test_composition in tqdm(test_compositions):
                my_logger.info(f"Proccesing {test_composition=}")
                test_composition_mask = [True if composition_idx_ == test_composition else False for composition_idx_ in composition_idx]
                chromas_composition = X[test_composition_mask]
                time_shifts_composition = time_shifts[test_composition_mask]

                composition_labeling = harmony_seq_pipeline(chromas_composition, time_shifts_composition, nochord_model, root_model, mode_model)
                composition_labeling = [[float(piece_begin), float(piece_end), int(harmony)] for (piece_begin, piece_end, harmony) in composition_labeling]

                gt_labeling = [[float(piece_begin), float(piece_end), int(harmony)] for (piece_begin, piece_end, harmony) in y[test_composition_mask][:, 1:]]

                try:
                    score = calc_normalized_intesection_drop_nochord(gt_labeling, composition_labeling)
                except Exception as e:
                    my_logger.info(f'Problem calc score...\n{e=}')
                    cnt_tracks_wo_chord += 1
                    continue
                    
                labeling[int(test_composition)] = {
                    'gt_labeling': gt_labeling,
                    'pred_labeling': composition_labeling,
                    'score': score,
                }

                score_sum += score

                
                with open(LABELING_FILE, 'w') as f:
                    json.dump(labeling, f)

            my_logger.info(f'Fold {i}')
            my_logger.info(f'Total score {score_sum / (len(test_compositions) - cnt_tracks_wo_chord)}')


# n = 10, 9 min (1 test sample)
# Total score 0.8136181575761506
# with negative
# Total score 0.8245660881264376
