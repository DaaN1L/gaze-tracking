from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


class ScreenPointEstimator:
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
        return mse(y, y_pred)

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
