import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

win_min = 0.5
win_max = 2.5

SEMITONES = 12
N_SAMPLES = 100_000
# DATA_DIR = "/data/autochord"
# DATA_DIR = "/storage/naumtsevalex/harmony/data/castom_npy"
LOAD_DATA_DIR = "/storage/naumtsevalex/harmony/src/data/root_McGill-Billboard/glinka_chord/npy"
SAVE_DATA_DIR = "/storage/naumtsevalex/harmony/src/data/root_McGill-Billboard/glinka_chord/castom_npy"



def merge_chords(X, y):
    X_out = []
    y_out = []
    chromas = X[0].copy()
    cnt = 1
    for label_ind in tqdm(range(1, len(y))):
        if y[label_ind] == y[label_ind - 1]:
            chromas += X[label_ind]
            cnt += 1
        else:
            X_out.append(chromas / cnt)
            chromas = X[label_ind].copy()
            cnt = 1
            y_out.append(y[label_ind - 1])

        if label_ind + 1 == len(y):
            X_out.append(chromas / cnt)
            y_out.append(y[label_ind])
    return np.array(X_out), np.array(y_out)

def create_negative_root_chord(X, y):
    X_out = []
    y_out = []
    # chromas = X[0].copy()
    # cnt = 1
    for label_ind in tqdm(range(0, len(y) - 1)):
        # 0. just sum
        chromas = (X[label_ind].copy() + X[label_ind + 1].copy()) / 2.
        
        # 1. random mask
        # mask = (np.random.random((len(X[label_ind]),)) > 0.3).astype(dtype=np.float32)
        # last_chromas = X[label_ind].copy() * mask
        # cur_chromas = X[label_ind + 1].copy()
        # sum_freq = np.sum(last_chromas) + cur_chromas + np.finfo(float).eps
        # chromas = (last_chromas + cur_chromas) / sum_freq
        
        # 2. last * alpha
        # alpha = 0.6
        # last_chromas = X[label_ind].copy() * alpha
        # cur_chromas = X[label_ind + 1].copy()
        # sum_freq = np.sum(last_chromas) + cur_chromas + np.finfo(float).eps
        # chromas = (last_chromas + cur_chromas) / sum_freq
        
        
        X_out.append(chromas)
        # maybe y[label_ind] == 0 and y[label_ind + 1] i find this isnt exsitsed root mode       
        y_out.append(0)

    return np.array(X_out), np.array(y_out)
    

def transform_into_dataset_of_modes(X, y):
    X_out = []
    y_out = []

    for i, item in enumerate(tqdm(X)):
        if y[i] > 0:
            root = (y[i] - 1) % 12
            mode = 1 if y[i] <= 12 else 0
            out_vec = np.zeros(SEMITONES * 2)
            out_vec[:SEMITONES] = np.roll(X[i][:SEMITONES], -root)
            out_vec[SEMITONES:] = np.roll(X[i][SEMITONES:], -root)
            X_out.append(out_vec.copy())
            y_out.append(mode)
    return np.array(X_out), np.array(y_out)


def transform_into_dataset_of_shifts(X, y):
    X_out = np.zeros((X.shape[0] * SEMITONES, X.shape[1]))
    y_out_nochord = np.zeros((y.shape[0] * SEMITONES))
    y_out_chord = np.zeros((y.shape[0] * SEMITONES))
    y_out_major = np.zeros((y.shape[0] * SEMITONES))

    for i, (X_entry, y_entry) in enumerate(tqdm(zip(X, y), total=X.shape[0])):
        start_ind = i * SEMITONES
        for shift in range(SEMITONES):
            X_out[start_ind + shift][:SEMITONES] = np.roll(X_entry[:SEMITONES], -shift)
            X_out[start_ind + shift][SEMITONES:] = np.roll(X_entry[SEMITONES:], -shift)

            if y_entry == 0:
                y_out_nochord[start_ind + shift] = 1
                y_out_chord[start_ind + shift] = 0
                y_out_major[start_ind + shift] = 0
            else:
                y_out_nochord[start_ind + shift] = 0
                actual_chord = (y_entry - 1) % 12  # account for N and maj/min division
                actual_mode = 1 if y_entry <= 12 else 0

                # this time classifiyng for A (not C)
                if actual_chord == shift:
                    y_out_chord[start_ind + shift] = 1
                if actual_mode == 1:
                    y_out_major[start_ind + shift] = 1
    return X_out, y_out_chord, y_out_major, y_out_nochord


def get_no_chord_positives(X, labels, composition_indices, timeshifts, selected_composition_indices, num_samples):
    assert len(X) == len(labels)
    assert len(X) == len(composition_indices)
    assert len(X) == len(timeshifts)

    X_out = []
    y_out = []
    max_retry_count = 10

    sample_indices = np.random.choice(selected_composition_indices, size=num_samples)
    
    for composition_id in tqdm(sample_indices):
        cur_retry_count = 0
        mask = composition_indices == composition_id
        X_selected = X[mask]
        labels_selected = labels[mask]
        timeshifts_selected = timeshifts[mask]

        while cur_retry_count < max_retry_count:
            window_size = np.random.uniform(win_min, 1.0)
            window_begin = np.random.choice(len(X_selected))

            cur_idx = window_begin
            cur_piece = X_selected[cur_idx].copy()
            cur_count = 1
            cur_labels = set()
            cur_labels.add(labels_selected[cur_idx])
            cur_window_size = timeshifts_selected[cur_idx, 1] - timeshifts_selected[cur_idx, 0]
            retry = False

            while cur_window_size < window_size:
                cur_idx += 1

                if cur_idx == len(X_selected):
                    retry = True
                    break

                cur_window_size += timeshifts_selected[cur_idx, 1] - timeshifts_selected[cur_idx, 0]
                cur_piece += X_selected[cur_idx]
                cur_labels.add(labels_selected[cur_idx])
                cur_count += 1
            
            if len(cur_labels) == 1 or retry:
                cur_retry_count += 1
                continue
            else:
                X_out.append(cur_piece / cur_count)
                y_out.append(1)
                break

    return np.array(X_out), np.array(y_out)


def load_np(filename, data_dir):

    if len(filename.split('.')) < 2:
        return np.load(os.path.join(data_dir, filename + '.npy'))
    return np.load(os.path.join(data_dir, filename))


def save_np(filename, array, data_dir):
    
    if len(filename.split('.')) < 2:
        np.save(os.path.join(data_dir, filename + ".npy"), array)
    else:
        np.save(os.path.join(data_dir, filename), array)

if __name__ == "__main__":
    X = load_np("01_all_chroma_vectors.npy", data_dir=LOAD_DATA_DIR)
    y = load_np("01_all_chord_labels.npy", data_dir=LOAD_DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X[:N_SAMPLES], y[:N_SAMPLES], train_size=0.8, random_state=10
    )

    X_train_modes, y_train_modes = transform_into_dataset_of_modes(
        X_train, y_train
    )
    X_test_modes, y_test_modes = transform_into_dataset_of_modes(
        X_test, y_test
    )
    (
        X_train_shifts,
        y_train_chord,
        _,
        _,
    ) = transform_into_dataset_of_shifts(X_train, y_train)
    (
        X_test_shifts,
        y_test_chord,
        _,
        _,
    ) = transform_into_dataset_of_shifts(X_test, y_test)

    X_merged, y_merged = merge_chords(X, y)
    X_train_merged, X_test_merged, y_train_merged, y_test_merged = train_test_split(
        X_merged, y_merged, train_size=0.8, random_state=10
    )

    X_train_modes_merged, y_train_modes_merged = transform_into_dataset_of_modes(
        X_train_merged, y_train_merged
    )
    X_test_modes_merged, y_test_modes_merged = transform_into_dataset_of_modes(
        X_test_merged, y_test_merged
    )
    (
        X_train_shifts_merged,
        y_train_chord_merged,
        _,
        _,
    ) = transform_into_dataset_of_shifts(X_train_merged, y_train_merged)

    (
        X_test_shifts_merged,
        y_test_chord_merged,
        _,
        _,
    ) = transform_into_dataset_of_shifts(X_test_merged, y_test_merged)

    filenames_to_arrays = {
        "X_train": X_train,
        "y_train": y_train,

        "X_test": X_test,
        "y_test": y_test,

        "X_train_modes": X_train_modes,
        "y_train_modes": y_train_modes,

        "X_test_modes": X_test_modes,
        "y_test_modes": y_test_modes,

        "X_train_shifts": X_train_shifts,
        "y_train_chord": y_train_chord,

        "X_test_shifts": X_test_shifts,
        "y_test_chord": y_test_chord,

        "X_merged": X_merged,
        "y_merged": y_merged,

        "X_train_merged": X_train_merged,
        "y_train_merged": y_train_merged,

        "X_test_merged": X_test_merged,
        "y_test_merged": y_test_merged,

        "X_train_modes_merged": X_train_modes_merged,
        "y_train_modes_merged": y_train_modes_merged,

        "X_test_modes_merged": X_test_modes_merged,
        "y_test_modes_merged": y_test_modes_merged,

        "X_train_shifts_merged": X_train_shifts_merged,
        "y_train_chord_merged": y_train_chord_merged,

        "X_test_shifts_merged": X_test_shifts_merged,
        "y_test_chord_merged": y_test_chord_merged,
}
    for filename, array in filenames_to_arrays.items():
        save_np(filename, array, data_dir=SAVE_DATA_DIR)
