from typing import Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
import xgboost

from hamilton.function_modifiers import extract_fields, config, load_from, value


@extract_fields(dict(
    X=np.ndarray,
    y=np.ndarray,
))
@load_from.parquet(path=value("./data/preprocessed_dataset.parquet"), inject_="dataset_parquet")
def dataset(dataset_parquet: pd.DataFrame, target: str) -> dict:
    """Preprocessed dataset ready for ML training; result from features.preprocessed_dataset"""
    features_mask = ~dataset_parquet.columns.isin(["days_due_mean", "days_due_std"])
    return dict(
        X=dataset_parquet.loc[:, features_mask].to_numpy(),
        y=dataset_parquet[target].to_numpy(),
    )


@config.when(splits_method="users")
def data_splits(X: np.ndarray) -> list[tuple]:
    kfold = KFold(n_splits=5)
    splits = [(train_idx, test_idx) for train_idx, test_idx in kfold.split(X)]
    return splits


@config.when(model_type="xgboost")
def model_config(model_config_override: Optional[dict] = None) -> dict:
    config = dict(
        booster="gbtree",
        learning_rate=0.05,  # alias: eta; typical 0.01 to 0.2
        max_depth=3,  # typical 3 to 10; will lead to overfitting
        gamma=0.1,  # alias: min_split_loss; 0 to +inf
        n_estimators=200,
        colsample_bytree=1,  # typical 0.5 to 1
        subsample=1,  # typical 0.6 to 1
        min_child_weight=1,  # 0 to +inf; prevent overfitting; too high underfit
        max_delta_step=0,  # 0 is no constraint; used in imbalanced logistic reg; typical 1 to 10;
        reg_alpha=0,  # alias alpha; default 0
        reg_lambda=1,  # alias lambda; default 1
        tree_method="gpu_hist",
        enable_categorical=True,
        max_cat_to_onehot=None,
        # learning parameters
        objective="reg:squarederror",
        # eval_metric="rmse",  # xgboost handles sensible defaults for builtin objectives
        seed=0,
        # others
        verbosity=2,  # 0: silent, 1: warning, 2: info, 3: debug
        callbacks=None,
    )
    if model_config_override:
        config.update(**model_config)
    return config


# def optuna_study(optuna_config: Optional[dict] = None) -> optuna.study.Study:
#     default = dict(
#         direction="maximize",
#         n_trials=10,
#         n_folds=3,
#         distributions=dict(
#             n_estimators=optuna.distributions.IntDistribution(250, 700, step=150),
#             learning_rate=optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
#             max_depth=optuna.distributions.FloatDistribution(3, 10),
#             gamma=optuna.distributions.FloatDistribution(0.01, 20, log=True),
#             colsample_bytree=optuna.distributions.FloatDistribution(0.6, 1),
#             min_child_weight=optuna.distributions.IntDistribution(1, 20, log=True),
#             max_delta_step=optuna.distributions.IntDistribution(0, 10),
#         )
#     )
#     if optuna_config:
#         default.update(**optuna_config)
#     return optuna.create_study(**default)  


@config.when(model_type="xgboost")
def validation_pred(
    X: np.ndarray,
    y: np.ndarray,
    data_splits: list[tuple],
    model_config: dict,
    # optuna_study: optuna.study.Study,
) -> np.ndarray:
    test_predictions = []
    for split_idx, (train_idx, test_idx) in enumerate(data_splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        model = xgboost.XGBRegressor(**model_config)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        test_predictions.append(pred)


    return np.concatenate(test_predictions)


def cross_validation_eval(
    y: np.ndarray,
    validation_pred: np.ndarray,
    scorer_names: list[str]
) -> dict:
    """
    ref: https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    """
    results = {}
    for scorer_name in scorer_names:
        scorer = get_scorer(scorer_name)
        score = scorer._score_func(y, validation_pred)
        results[scorer_name] = score
    return results


@config.when(model_type="xgboost")
def trained_model(X: np.ndarray, y: np.ndarray, model_config: dict) -> xgboost.XGBRegressor:
    model = xgboost.XGBRegressor(**model_config)
    model.fit(X, y)
    return model


@config.when(model_type="xgboost")
def save_trained_model(
    trained_model: xgboost.XGBRegressor,
    trained_model_path: str,
) -> bool:
    trained_model.save_model(trained_model_path)
    return trained_model
    