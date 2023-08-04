import numpy as np
import pandas as pd


def signups() -> pd.Series:
    """Returns sign up values"""
    return pd.Series([1, 10, 50, 100, 200, 400])


def spend() -> pd.Series:
    """Returns the spend values"""
    return pd.Series([10, 10, 20, 40, 40, 50])


def spend_per_signup(spend: pd.Series, signups: pd.Series) -> pd.Series:
    """The cost per signup in relation to spend."""
    return spend / signups


def log_spend_per_signup(spend_per_signup: pd.Series) -> pd.Series:
    """Simple function taking the logarithm of spend over signups."""
    return np.log(spend_per_signup)