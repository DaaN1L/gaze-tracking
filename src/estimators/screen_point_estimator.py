from abc import ABC, abstractmethod

from catboost import CatBoostRegressor, Pool, FeaturesData

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


class ScreenPointEstimator(ABC):
    @abstractmethod
    def fit(self, df, target_columns: list, train_size=0.7):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @property
    @abstractmethod
    def train_mse(self):
        ...

    @property
    @abstractmethod
    def val_mse(self):
        ...


class ScreenPointEstimatorSklearn(ScreenPointEstimator):
    def __init__(self, base_estimator, seed=None, params=None):
        if params is None:
            params = {}
        self.seed = seed
        reg = MultiOutputRegressor(base_estimator(random_state=self.seed, **params), n_jobs=-1)
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()),
             ("regressor", reg)]
        )
        self._train_mse = 0
        self._val_mse = 0

    def _score(self, X, y):
        y_pred = self.pipeline.predict(X)
        return mse(y, y_pred, squared=False)

    def fit(self, df, target_columns: list, train_size=0.7):
        X, y = df.drop(columns=target_columns), df[target_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.seed, train_size=train_size)
        self.pipeline.fit(X_train.values, y_train.values)
        self._train_mse = self._score(X_train, y_train)
        self._val_mse = self._score(X_test, y_test)

    def predict(self, X):
        return self.pipeline.predict(X)

    @property
    def train_mse(self):
        return self._train_mse

    @property
    def val_mse(self):
        return self._val_mse


class ScreenPointEstimatorCatboost(ScreenPointEstimator):
    # use_best_model
    # early_stopping_rounds
    # l2_leaf_reg
    # random_strength
    def __init__(self, **params):
        self.seed = params.get("random_seed", None)
        params["loss_function"] = "MultiRMSE"
        params["eval_metric"] = "MultiRMSE"

        self.regressor = CatBoostRegressor(random_state=self.seed, **params)
        self._train_mse = 0
        self._val_mse = 0

    @staticmethod
    def _make_data_pool(X, y=None):
        return Pool(
            data=FeaturesData(
                num_feature_data=np.asanyarray(X, dtype=np.float32)
            ),
            label=y
        )

    def fit(self, df, target_columns: list, train_size=0.7):
        X, y = df.drop(columns=target_columns), df[target_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.seed, train_size=train_size)
        train_data = self._make_data_pool(X_train.values, y_train.values)
        test_data = self._make_data_pool(X_test.values, y_test.values)
        self.regressor.fit(X=train_data, eval_set=test_data)
        best_iter = self.regressor.best_iteration_
        self._train_mse = self.regressor.evals_result_["learn"]["MultiRMSE"][best_iter]
        self._val_mse = self.regressor.evals_result_["validation"]["MultiRMSE"][best_iter]

    def predict(self, X):
        X_ = self._make_data_pool(X)
        return self.regressor.predict(X_)

    @property
    def train_mse(self):
        return self._train_mse

    @property
    def val_mse(self):
        return self._val_mse
