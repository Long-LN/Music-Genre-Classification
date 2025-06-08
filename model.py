from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import numpy as np


class Model:

    def __init__(self, feature_matrix, labels_tuple, cfg):
        self.X = feature_matrix

        genre_labels, mood_labels = labels_tuple

        self.encoder_genre = LabelEncoder()
        self.encoder_mood = LabelEncoder()

        self.y_genre = self.encoder_genre.fit_transform(genre_labels)
        self.y_mood = self.encoder_mood.fit_transform(mood_labels)

        self.cfg = cfg
        self.best_estimator = None
        self.holdout_test_set = None
        self.holdout_val_set = None

    def train_kfold(self):
        # Tách dữ liệu với nhãn genre làm stratify chính
        X_cv, X_test, y_cv_genre, y_test_genre, y_cv_mood, y_test_mood = train_test_split(
            self.X,
            self.y_genre,
            self.y_mood,
            random_state=42,
            stratify=self.y_genre,
            **self.cfg['tt_test_dict']
        )
        self.holdout_test_set = (X_test, (y_test_genre, y_test_mood))

        X_train, X_val, y_train_genre, y_val_genre, y_train_mood, y_val_mood = train_test_split(
            X_cv,
            y_cv_genre,
            y_cv_mood,
            random_state=42,
            stratify=y_cv_genre,
            **self.cfg['tt_val_dict']
        )
        self.holdout_val_set = (X_val, (y_val_genre, y_val_mood))

        # Pipeline với MultiOutputClassifier
        pipe = Pipeline([
            ('scaler', self.cfg['scaler']),
            ('model', MultiOutputClassifier(self.cfg['base_model']))
        ])

        # Chuyển param_grid sang dạng MultiOutputClassifier
        param_grid = {}
        for k, v in self.cfg['param_grid'].items():
            if k.startswith('model__'):
                param_grid[k.replace('model__', 'model__estimator__')] = v
            else:
                param_grid[k] = v

        if 'iid' in self.cfg['grid_dict']:
            del self.cfg['grid_dict']['iid']

        # ✅ Sử dụng KFold thay cho StratifiedKFold vì nhãn là đa đầu ra
        kf = KFold(**self.cfg['kf_dict'])

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=kf,
            return_train_score=True,
            **self.cfg['grid_dict']
        )

        # Kết hợp nhãn: y = [[genre, mood], ...]
        y_train_multi = np.vstack([y_train_genre, y_train_mood]).T

        grid_search.fit(X_train, y_train_multi)
        self.best_estimator = grid_search.best_estimator_

    def _parse_conf_matrix(self, cnf_matrix):
        TP = np.diag(cnf_matrix)
        FP = cnf_matrix.sum(axis=0) - TP
        FN = cnf_matrix.sum(axis=1) - TP
        TN = cnf_matrix.sum() - (FP + FN + TP)

        return TP.astype(float), FP.astype(float), TN.astype(float), FN.astype(float)

    def _predict(self, holdout_type):
        if holdout_type == "val":
            X_holdout, (y_holdout_genre, y_holdout_mood) = self.holdout_val_set
        elif holdout_type == "test":
            X_holdout, (y_holdout_genre, y_holdout_mood) = self.holdout_test_set
        else:
            raise ValueError("holdout_type phải là 'val' hoặc 'test'")

        scaler = self.best_estimator['scaler']
        model = self.best_estimator['model']

        X_holdout_scaled = scaler.transform(X_holdout)
        y_pred = model.predict(X_holdout_scaled)  # y_pred.shape = (n_samples, 2)

        # Confusion matrix cho từng đầu ra
        cnf_genre = confusion_matrix(y_holdout_genre, y_pred[:, 0])
        cnf_mood = confusion_matrix(y_holdout_mood, y_pred[:, 1])

        TP_g, FP_g, TN_g, FN_g = self._parse_conf_matrix(cnf_genre)
        TP_m, FP_m, TN_m, FN_m = self._parse_conf_matrix(cnf_mood)

        return (TP_g, FP_g, TN_g, FN_g), (TP_m, FP_m, TN_m, FN_m)

    def predict(self, holdout_type):
        (TP_g, FP_g, TN_g, FN_g), (TP_m, FP_m, TN_m, FN_m) = self._predict(holdout_type)

        print(f'{holdout_type} Set - Genre per class:')
        print(f'TP:{TP_g}, FP:{FP_g}, TN:{TN_g}, FN:{FN_g}')
        print(f'{holdout_type} FPR (Genre): {FP_g / (FP_g + TN_g)}')
        print(f'{holdout_type} FNR (Genre): {FN_g / (TP_g + FN_g)}')
        print(f'{holdout_type} Accuracy (Genre): {(TP_g + TN_g) / (TP_g + TN_g + FP_g + FN_g)}')

        print(f'\n{holdout_type} Set - Mood per class:')
        print(f'TP:{TP_m}, FP:{FP_m}, TN:{TN_m}, FN:{FN_m}')
        print(f'{holdout_type} FPR (Mood): {FP_m / (FP_m + TN_m)}')
        print(f'{holdout_type} FNR (Mood): {FN_m / (TP_m + FN_m)}')
        print(f'{holdout_type} Accuracy (Mood): {(TP_m + TN_m) / (TP_m + TN_m + FP_m + FN_m)}')
