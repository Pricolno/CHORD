import numpy as np

from catboost import CatBoostClassifier
from tqdm import tqdm

from preprocess_data import load_np, SEMITONES


SAMPLES = 250

PARAMS_ROOT = {
    "loss_function": "CrossEntropy",
    "n_estimators": 300,
    "learning_rate": 0.1,
    "depth": 10,
    "l2_leaf_reg": 62.37934704452352,
    "bootstrap_type": "Bayesian",
    "random_strength": 1.1488785147899944e-08,
    "bagging_temperature": 1.5481239273448555,
    "od_type": "Iter",
    "od_wait": 49,
    "random_seed": 10,
}
PARAMS_MODE = {
"loss_function": "CrossEntropy",
    "n_estimators": 300,
    "learning_rate": 0.07,
    "depth": 10,
    "l2_leaf_reg": 0.10285856054142818,
    "bootstrap_type": "Bayesian",
    "random_strength": 0.0025811780488595813,
    "bagging_temperature": 0.13879907325154095,
    "od_type": "IncToDec",
    "od_wait": 28,
    "random_seed": 10,
}
PARAMS_NOCHORD = {
"loss_function": "CrossEntropy",
    "n_estimators": 400,
    "learning_rate": 0.1,
    "depth": 10,
    "l2_leaf_reg": 62.37934704452352,
    "bootstrap_type": "Bayesian",
    "random_strength": 1.1488785147899944e-08,
    "bagging_temperature": 1.5481239273448555,
    "od_type": "Iter",
    "od_wait": 49,
    "random_seed": 10,
}


def get_accuracy(model_roots, model_mode, model_nochord, X_merged, X_shifts, X_modes, y):
    preds_root = model_roots.predict_proba(X_shifts, verbose=True)
    # preds_mode = model_mode.predict_proba(X_modes, verbose=True)
    preds_nochord = model_nochord.predict(X_merged, verbose=True)

    correct = 0
    corr_nochord = 0
    for i in tqdm(range(0, len(X_shifts), 12), total=len(X_shifts) // 12):
        nochord = 1 if y[i // 12] == 0 else 0  # only 0.01% of Ns are correctly predicted -> now 0.2% on train 0.1% on test
        if nochord == preds_nochord[i // 12] == 1:
            corr_nochord += 1
            correct += 1
        elif nochord == preds_nochord[i // 12] == 0:
            preds = preds_root[i : i + 12]
            predicted_root = np.argmax(preds, axis=0)[1]
            shifted_by_pred_root = np.zeros(SEMITONES * 2)
            shifted_by_pred_root[:SEMITONES] = np.roll(X_merged[i // 12, :SEMITONES], -predicted_root)
            shifted_by_pred_root[SEMITONES:] = np.roll(X_merged[i // 12, SEMITONES:], -predicted_root)
            predicted_mode = np.argmax(model_mode.predict_proba(shifted_by_pred_root))
            
            if predicted_mode == 0:
                predicted_root += 12
            final_pred = predicted_root + 1
            # preds_vector = np.zeros(24)
            # for j in range(12):
            #     preds_vector[j] = preds[j, 1] * preds_mode[j, 1]
            # for j in range(12, 24):
            #     preds_vector[j] = preds[j - 12, 1] * preds_mode[j - 12, 0]
            # final_pred = np.argmax(preds_vector) + 1
            if final_pred == y[i // 12]:
                correct += 1
    print(f"CORRECTLY PREDICTED Ns: {corr_nochord} ({corr_nochord / len(X_modes)*100:.2f}%")
    return correct / len(X_modes)


def train():
    X_train_merged = load_np("X_train")
    y_train_merged = load_np("y_train")

    X_test_merged = load_np("X_test")
    y_test_merged = load_np("y_test")

    X_train_shifts_merged = load_np("X_train_shifts")
    y_train_chord_merged = load_np("y_train_chord")

    X_test_shifts_merged = load_np("X_test_shifts")
    # y_test_chord_merged = np.load("y_test_chord_merged")

    X_train_modes_merged = load_np("X_train_modes")
    y_train_modes_merged = load_np("y_train_modes")

    X_test_modes_merged = load_np("X_test_modes")
    # y_test_modes_merged = np.load("y_test_modes_merged")

    y_train_nochord_merged = np.where(y_train_merged == 0, 1, 0)
    y_test_nochord_merged = np.where(y_test_merged == 0, 1, 0)

    model_roots = CatBoostClassifier(**PARAMS_ROOT, logging_level="Verbose")
    model_mode = CatBoostClassifier(**PARAMS_MODE, logging_level="Verbose")
    model_nochord = CatBoostClassifier(**PARAMS_NOCHORD, logging_level="Verbose")

    model_roots.fit(X_train_shifts_merged, y_train_chord_merged)
    model_mode.fit(X_train_modes_merged, y_train_modes_merged)
    model_nochord.fit(X_train_merged, y_train_nochord_merged) # TODO: w/o any shifts

    acc_train = get_accuracy(
        model_roots,
        model_mode,
        model_nochord,
        X_train_merged,
        X_train_shifts_merged,
        X_train_modes_merged,
        y_train_merged,
    )
    acc_test = get_accuracy(
        model_roots,
        model_mode,
        model_nochord,
        X_test_merged,
        X_test_shifts_merged,
        X_test_modes_merged,
        y_test_merged,
    )
    print(f"TRAIN ACCURACY: {acc_train}")
    print(f"TEST ACCURACY: {acc_test}")
    # TRAIN ACCURACY: 0.8051621068504955
    # TEST ACCURACY: 0.7338137233813723

if __name__ == "__main__":
    train()
