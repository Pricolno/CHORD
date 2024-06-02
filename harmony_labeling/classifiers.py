import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


class BaseClassifier:
    """Classifier for root or mode."""
    def __init__(self, params, used_features) -> None:
        """Initializes CatBoost. 
        We need to specify used features, their columns in DataFrame and preprocessing methods.
        """
        self.classifier = CatBoostClassifier(**params)
        # self.classifier = RandomForestClassifier()

        self.used_features = used_features

        self.features_columns = {
            'pitches': [f'pitch_{i}' for i in range(1, 13)],
            'shift': ['shift'],
            'mode': ['mode'],
            'meter': ['meter'],

            'root_probas': [f'root_proba_{i}' for i in range(12)],
        }

        self.column_transformers = {
            'pitches': ColumnTransformer([
                ('scaling', StandardScaler(), self.features_columns['pitches']),
            ]),
            'shift': ColumnTransformer([
                ('ohe', OneHotEncoder(handle_unknown='ignore',
                                    #   sparse=False
                                      ), self.features_columns['shift']),
            ]),
            'mode': ColumnTransformer([
                ('other',  'passthrough', self.features_columns['mode'])
            ]),
            'meter': ColumnTransformer([
                ('ohe', OneHotEncoder(handle_unknown='ignore',
                                    #   sparse=False
                ), self.features_columns['meter']),
            ]),
            
            'root_probas': ColumnTransformer([
                ('identity', FunctionTransformer(lambda x: x), self.features_columns['root_probas']),
            ]),
            
        }

        self.training = True
    
    def feature_preprocess(self, feature_name, feature_data) -> np.ndarray:
        if self.training:
            self.column_transformers[feature_name].fit(feature_data)
        return self.column_transformers[feature_name].transform(feature_data)
    
    def train(self, features: pd.DataFrame, target: np.ndarray) -> None:
        """Trains transformers and classifier."""
        self.training = True
        processed_features = []
        for feature in self.used_features:
            processed_feature = self.feature_preprocess(feature, features[self.features_columns[feature]])
            processed_features.append(processed_feature)
        X = np.hstack(processed_features)
        y = target
        self.classifier.fit(X, y)
    
    def pred(self, features: pd.DataFrame) -> np.ndarray:
        """Predicts values.
        Returns labels and probas.
        """
        self.training = False
        processed_features = []
        for feature in self.used_features:
            processed_feature = self.feature_preprocess(feature, features[self.features_columns[feature]])
            processed_features.append(processed_feature)
        X = np.hstack(processed_features)
        return self.classifier.predict(X), self.classifier.predict_proba(X)
    
