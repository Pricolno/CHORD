import numpy as np
import pandas as pd
from scipy.linalg import circulant
from sklearn.model_selection import KFold

from classifiers import BaseClassifier

from collections import Counter

from metrics import calculate_bach_metrics, calculate_WSCR

keys_tokenizer = {
    'C': 0, 
    'C#': 1,
    'D': 2,
    'D#': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G': 7,
    'G#': 8,
    'A': 9,
    'A#': 10,
    'B': 11
}

# USE_GPU = True
USE_GPU = False

GPU_SETTINGS = {
    'task_type': "GPU",
    'devices': '0'
}


best_params_roots = {
    'n_estimators': 685,
    # 'n_estimators': 10,
    'learning_rate': 0.07557142211700456,
    'depth': 7,
    'l2_leaf_reg': 62.37934704452352,
    'bootstrap_type': 'Bayesian',
    'random_strength': 1.1488785147899944e-08,
    'bagging_temperature': 1.5481239273448555,
    'od_type': 'Iter',
    'od_wait': 49,
    'eval_metric': 'Accuracy',
    'verbose': False
}

best_params_mode = {
    'n_estimators': 290,
    # 'n_estimators': 10,
    'learning_rate': 0.04774974078860072,
    'depth': 10,
    'l2_leaf_reg': 0.10285856054142818,
    'bootstrap_type': 'Bayesian',
    'random_strength': 0.0025811780488595813,
    'bagging_temperature': 0.13879907325154095,
    'od_type': 'IncToDec',
    'od_wait': 28,
    'verbose': False
}

if USE_GPU:
    best_params_roots.update(GPU_SETTINGS)
    best_params_mode.update(GPU_SETTINGS)


def tokenize_mode(mode_col: pd.Series) -> np.ndarray:
    return (mode_col == 'M').astype('int32').to_numpy()

def tokenize_root(root_col: pd.Series) -> np.ndarray:
    return root_col.apply(lambda x: keys_tokenizer[x]).to_numpy()

class HarmonyClassifier:
    """Mode and root classifier."""
    def __init__(self) -> None:
        """Inits classifiers and features columns map."""
        self.root_classifier = BaseClassifier(best_params_roots, ['pitches', 'mode'])
        self.mode_classifier = BaseClassifier(best_params_mode, ['pitches'])
        self.features_columns = {
            'pitches': [f'pitch_{i}' for i in range(1, 13)],
            'root': ['root'],
            'mode': ['mode']
        }

    def apply_shifts(self, data: pd.DataFrame):
        """Augments data for root classification."""
        data = data.reset_index()
        
        tonic_features = np.zeros((12 * data.shape[0], 12))
        tonic_labels = np.zeros((12 * data.shape[0]))
        res_mode = np.zeros((12 * data.shape[0]))

        pitches = data[self.features_columns['pitches']].to_numpy()
        # roots = tokenize_root(data['root'])
        roots = data['root'].to_numpy()
        mode = data['mode'].to_numpy()
        print(Counter(roots))
        print(Counter(mode))
        
        for i in range(pitches.shape[0]):
            pitches_shifted = np.roll(pitches[i, :], -roots[i])
            tonic_features[i * 12:(i + 1) * 12, :] = circulant(pitches_shifted)
            tonic_labels[i * 12] = 1
            res_mode[i * 12:(i + 1) * 12] = mode[i]

        res = pd.DataFrame(data=tonic_features, columns=self.features_columns['pitches'])
        res['label'] = tonic_labels
        res['mode'] = res_mode
        return res
    
    def filter_data(self, data: pd.DataFrame):
        """Filters rare chords. Not used."""
        cols = 'chord_label'
        counts = data.groupby(by=cols).size().reset_index().rename(columns={0: 'count'})
        filtered_df = data.merge(counts[counts['count'] >= 5], left_on=cols, right_on=cols)
        return filtered_df

    def train(self, data: pd.DataFrame) -> None:
        """Trains mode and root classifiers.
        Mode feature is used as mode_classifier's inference.
        """
        # print(data)
        self.mode_classifier.train(
            data,
            data['mode']
        )
        
        shifted_data = self.apply_shifts(data)
        labels = shifted_data['label'].to_numpy()
        self.root_classifier.train(
            shifted_data, 
            labels
        )

    def pred(self, data: pd.DataFrame) -> np.ndarray:
        """Predicts root and mode."""
        data = data.copy()
        
        mode, _ = self.mode_classifier.pred(data)
        data['mode'] = mode
        # data['root'] = '0'
        data['root'] = 0
        shifted_data = self.apply_shifts(data)
        _, root_probas = self.root_classifier.pred(shifted_data)
        roots = root_probas[:, 1].reshape((data.shape[0], 12))
        roots = roots.argmax(axis=1)
        return roots, mode


class DoubleHarmonyClassifier(HarmonyClassifier):
    """Mode and root classifier."""
    def __init__(self):
        self.harmony_classifier = HarmonyClassifier()
        self.features_columns = {
            'pitches': [f'pitch_{i}' for i in range(1, 13)],
            'root_probas': [f'root_proba_{i}' for i in range(12)],
            'root': ['root'],
            'mode': ['mode']
        }
        self.root_classifier = BaseClassifier(best_params_roots, [
            'pitches',
            'root_probas',
            # 'mode'
        ])
        self.mode_classifier = BaseClassifier(best_params_roots, [
            'pitches',
            'root_probas'
        ])

    def train(self, data: pd.DataFrame) -> None:
        """Trains mode and root classifiers.
        Mode feature is used as mode_classifier's inference.
        """
        self.harmony_classifier.train(data)
        roots, mode = self.harmony_classifier.pred(data)        
        shifted_data = self.apply_shifts(data)
        
        _, root_probas = self.harmony_classifier.root_classifier.pred(shifted_data)
        root_probas = root_probas[:, 1].reshape((data.shape[0], 12))

        gt_modes = data['mode'].to_numpy()
        # data['mode'] = mode
        
        gt_roots = data['root'].to_numpy()
        # data['root'] = roots

        features = data[self.features_columns['pitches'] + self.features_columns['mode']].copy()
        features[self.features_columns['root_probas']] = root_probas
        self.mode_classifier.train(features, gt_modes)

        
        self.root_classifier.train(features, gt_roots)
        


    def pred(self, data: pd.DataFrame) -> np.ndarray:
        """Predicts root and mode."""
        roots, mode = self.harmony_classifier.pred(data)
        shifted_data = self.apply_shifts(data)
        _, root_probas = self.harmony_classifier.root_classifier.pred(shifted_data)
        root_probas = root_probas[:, 1].reshape((data.shape[0], 12))

        # data['mode'] = mode
        features = data[self.features_columns['pitches'] + self.features_columns['mode']].copy()
        features[self.features_columns['root_probas']] = root_probas
        # mode, _ = self.mode_classifier.pred(features)
        roots, _ = self.root_classifier.pred(features)
        roots = roots.squeeze(-1)
        
        return roots, mode
    

class BaseDataset:
    """Base class for harmony datasets.
    Requires DataFrame with pitches, mode and root for initialization.
    """
    def __init__(self, data: pd.DataFrame, use_double_harmony: bool = False) -> None:
        """Save data and initialize harmony classifier."""
        self.data = data
        if use_double_harmony:
            self.harmony_classifier = DoubleHarmonyClassifier()
        else:
            self.harmony_classifier = HarmonyClassifier()

    def get_accuracy(self, root_pred, mods_pred, roots, mode):
        """Get overall accuracy by predicted and gt roots and modes."""
        return ((root_pred == roots) & (mods_pred == mode)).sum() / root_pred.shape[0]
    
    def kfold_accuracy(self, mode='random', kfolds: list[int] = None):
        """Runs measuring accuracy by kfolds.
        mode: 
            'random' - chords are splitted randomly
            'separately' - chords are splitted by compositions
        """
        if mode == 'random':
            return self.kfold_accuracy_random()
        if mode == 'separately':
            return self.kfold_accuracy_separately(kfolds=kfolds)
        raise ValueError("mode not in ['random', 'separately']")

    def kfold_accuracy_separately(self, kfolds):
        accuracies = []
        kf = KFold(n_splits=10, random_state=100, shuffle=True)
        
        choral_ids = self.data['choral_ID'].unique()
        for cur_fold, (train_index, test_index) in enumerate(kf.split(choral_ids)):
            if kfolds is not None and cur_fold not in kfolds:
                continue
            
            train_chorales = set(choral_ids[train_index])
            test_chorales = set(choral_ids[test_index])

            train_data = self.data.loc[self.data['choral_ID'].apply(lambda x: x in train_chorales), :].copy()
            test_data = self.data.loc[self.data['choral_ID'].apply(lambda x: x in test_chorales), :].copy()

            accuracy = self.train_measure_accuracy(train_data, test_data)
            accuracies.append(accuracy)
        print('mean:', np.mean(accuracies))

    def kfold_accuracy_random(self):
        accuracies = []
        kf = KFold(n_splits=10, random_state=0, shuffle=True)
        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index, :].copy()
            test_data = self.data.loc[test_index, :]

            accuracy = self.train_measure_accuracy(train_data, test_data)
            accuracies.append(accuracy)
        print('mean:', np.mean(accuracies))

    def train_measure_accuracy(self, train_data: pd.DataFrame, test_data: pd.DataFrame):        
        train_data = train_data.copy()
        train_data.loc[:, 'root'] = train_data.loc[:, 'root'].apply(lambda x: keys_tokenizer[x]).to_numpy()
        train_data.loc[:, 'mode'] = train_data.loc[:, 'mode'].apply(lambda x: 1 if x == 'M' else 0).to_numpy()
        
        # print(f'{train_data=}')
        self.harmony_classifier.train(train_data)
        
        test_data = test_data.copy()
        test_data.loc[:, 'root'] = test_data.loc[:, 'root'].apply(lambda x: keys_tokenizer[x]).to_numpy()
        test_data.loc[:, 'mode'] = test_data.loc[:, 'mode'].apply(lambda x: 1 if x == 'M' else 0).to_numpy()

        # print(f'{test_data=}')
        root_pred, mode_pred = self.harmony_classifier.pred(
            test_data
        )
        
        # print(len(root_pred))
        # print(root_pred)
        
        # print(f'{root_pred.shape=}')
        # root_pred = root_pred.squeeze(-1)
        # print(f'{root_pred.shape=}')
        # print(f'{mode_pred.shape=}')
        
        # print(f'{test_data["root"].to_numpy().shape=}')
        # print(f'{test_data["mode"].to_numpy().shape=}')
        # metrics = calculate_bach_metrics(predicted_labels, true_labels)

        
        # accuracy = self.get_accuracy(
        #     root_pred,
        #     mode_pred, 
        #     test_data['root'].to_numpy(),
        #     test_data['mode'].to_numpy(),
        # )
        # print(accuracy)
        # HarmonyClassifier: 0.8166311300639659
        # mean: 0.814498933901919
        # DoubleHarmonyClassifier: 0.8315565031982942
        # 0.8315565031982942
        # 0.8400852878464818
        
        wscr = calculate_WSCR(
            # predicted_trackId_root_mode_list,
            [(track_id, r, m) for track_id, r, m in zip(test_data['choral_ID'].to_numpy(), root_pred, mode_pred)], 
            # gt_trackId_root_mode_list
            [(track_id, r, m) for track_id, r, m in zip(test_data['choral_ID'].to_numpy(), test_data['root'].to_numpy(), test_data['mode'].to_numpy())],
        )
        print(f'{wscr=}')
        
        
        
        
        # --- Calculate BaCh metrics ---
        full_chord_metrics = calculate_bach_metrics(
            # Combine root and mode predictions into a single list
            [(r, m) for r, m in zip(root_pred, mode_pred)],
            [(r, m) for r, m in zip(test_data['root'].to_numpy(), test_data['mode'].to_numpy())],
            evaluation_type="full_chord"
        )
        

        root_only_metrics = calculate_bach_metrics(
            root_pred,
            test_data['root'].to_numpy(),
            evaluation_type="root_only"
        )
        # ------------------------------

        # print("Accuracy:", accuracy)
        print("Full Chord Metrics:", full_chord_metrics)
        print("Root Only Metrics:", root_only_metrics)

        accuracy = full_chord_metrics['AccE']
        return accuracy

class chorales_dataset(BaseDataset):
    """Wrapper fo Bach Chorales dataset."""
    def __init__(self, csv_path: str, *args, **kwargs) -> None:
        self.csv_path = csv_path
        self.raw_features = pd.read_csv(csv_path)
        self.data = None
        self.create_features()
        super().__init__(self.data, *args, **kwargs)
    
    def create_features(self):
        """Transforming row features into features for classifier."""
        boundary_segments_count = 0
        data = []
        pitch_columns = [f'pitch_{i}' for i in range(1, 13)]

        for idx, row in self.raw_features.iterrows():
            row_data = [1 if row[pitch] == 'YES' else 0 for pitch in pitch_columns]
            # row_data = [elem / sum(row_data) for elem in row_data]
            root, mode = self.get_chord(row['chord_label'])
            if root is None:
                boundary_segments_count += 1
            else:
                row_data.append(row['choral_ID'])
                row_data.append(root)
                row_data.append(mode)
                row_data.append(row['chord_label'])
                row_data.append(row['meter'])
                data.append(row_data)
        columns = pitch_columns + ['choral_ID', 'root', 'mode', 'chord_label', 'meter']
        self.data = pd.DataFrame(data, columns=columns)

        order = ['choral_ID'] + pitch_columns + ['chord_label', 'root', 'mode', 'meter']
        self.data = self.data[order].copy()
        
        
        
        print('boundary segments:', boundary_segments_count)
        print('objects in dataset:', self.data.shape[0])

    def get_chord(self, label: str):
        """Transforming chord label into root note and mode."""
        if 'm' not in label.lower():
            return None, None
        if label[-1].isdigit():
            label = label[:-1]

        mode = label[-1]
        label = label[:-1]

        if label[-1] == '_':
            label = label[:-1]
        
        if 'b' in label:
            enharmonic_keys = {
                'Ab': 'G#',
                'Bb': 'A#',
                'Db': 'C#',
                'Eb': 'D#'
            }
            label = enharmonic_keys[label]

        return label, mode
