from typing import Optional

import xgboost
import optuna

from hamilton.function_modifiers import extract_fields, parameterize, config


@extract_fields(**dict(
    train_split=tuple,
    validation_split=tuple,
    test_split=tuple,
))
def create_splits(X_df: pd.DataFrame, y_df: pd.DataFrame) -> dict:
    return dict(
        train_split=...,
        validation_split=...,
        test_split=...,
    )


@config.when(model_type="xgboost")
def initialized_model(model_config: Optional[dict] = None) -> xgboost.XGBoostRegressor:
    default = dict(...)
    if model_config:
        default.update(**model_config)
    return xgboost.XGBoostRegressor(**default)


def optuna_study(optuna_config: Optional[dict] = None) -> optuna.study.Study:
    default = dict(...)
    if optuna_config:
        default.update(**optuna_config)
    return optuna.create_study(**default)


@extract_fields(**dict(
    trained_model=xgboost.XGBoostRegressor,
    completed_optuna_study=optuna.study.Study,
))
def train_model(
    initialized_model: xgboost.XGBoostRegressor,
    optuna_study: optuna.study.Study,
    train_split: tuple,
    validation_split: tuple,
) -> dict:
    return dict(
        trained_model=...,
        completed_optuna_study=...,
    )


@config.when(model_type="xgboost")
def save_trained_model(
    trained_model: xgboost.XGBoostRegressor,
    trained_model_path: str,
) -> bool:
    xgboost.save_model(trained_model_path)
    return True


@parameterize_sources(
    train_pred=dict(split="train_split"),
    validation_pred=dict(split="validation_split"),
    test_pred=dict(split="test_split"),
)
def predict(trained_model: xgboost.XGBoostRegressor, split: tuple) -> np.ndarray:
    return train_model.predict(split)