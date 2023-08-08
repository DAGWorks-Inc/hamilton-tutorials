import pandas as pd
import numpy as np

from hamilton.function_modifiers import extract_columns, extract_fields, load_from, save_to, value


CLEAN_APPLICATION_RECORD_COLUMNS = [
    'cnt_children', 'amt_income_total', 'days_birth', 'days_employed',
    'flag_mobil', 'flag_work_phone', 'flag_phone', 'flag_email',
    'cnt_fam_members', 'eduction_type_encoded', 'family_status_encoded',
    'gender_encoded', 'housing_type_encoded', 'income_type_encoded',
    'occupation_type_encoded', 'own_car_encoded', 'own_realty_encoded'
]

@extract_columns(*CLEAN_APPLICATION_RECORD_COLUMNS)
@load_from.parquet(path=value("./data/application_clean.parquet"))
def clean_application_record(clean_application_parquet: pd.DataFrame) -> pd.DataFrame:
    """Load the clean application records; results from data_cleaning.application_clean"""
    return clean_application_parquet


CLEAN_CREDIT_RECORD_COLUMNS = ['month_index', 'status']

@extract_columns(*CLEAN_CREDIT_RECORD_COLUMNS)
@load_from.parquet(path=value("./data/credit_clean.parquet"))
def clean_credit_record(clean_credit_parquet: pd.DataFrame) -> pd.DataFrame:
    """Load the clean credit records; results from data_cleaning.credit_clean"""
    return clean_credit_parquet


SALARIES_COLUMNS = ["all_salaries"]

@extract_columns(*SALARIES_COLUMNS)
@load_from.parquet(path=value("./data/salaries.parquet"))
def salaries_record(salaries_parquet: pd.DataFrame) -> pd.DataFrame:
    """Load salary data from external sources, with more users"""
    salaries_parquet = salaries_parquet.rename(columns={"salary": "all_salaries"})
    return salaries_parquet


def age_in_years(days_birth: pd.Series) -> pd.Series:
    """Convert days since birth into number of years"""
    return days_birth.abs().div(365).astype(int)


def age_binned(age_in_years: pd.Series) -> pd.Series:
    """Bin ages by decade; users must be at least 18"""
    bins = [18, 30, 40, 50, 60, 70, 80]
    binned_values = pd.cut(age_in_years, bins, right=False)
    return binned_values.cat.codes.astype(int)


@extract_fields(dict(
    income_bins=np.ndarray,
    income_percentile=pd.Series,
))
def income_percentile_(amt_income_total: pd.Series) -> dict:
    """Percentile of total income computed on the main dataset"""
    percentile, bins = pd.qcut(amt_income_total, 100, duplicates="drop", retbins=True)
    return dict(
        income_bins=bins,
        income_percentile=percentile.cat.codes.astype(int),
    )


@extract_fields(dict(
    salary_bins=np.ndarray,
    salary_percentile=pd.Series,
))
def salary_percentile_(clean_application_record: pd.DataFrame, all_salaries: pd.Series) -> dict:
    """Percentile of the salary computed on the external dataset"""
    percentile, bins = pd.qcut(all_salaries, 100, duplicates="drop", retbins=True)
    percentile_df = pd.Series(percentile.cat.codes, name="salary_percentile").to_frame()

    merged_df = pd.merge(
        clean_application_record, percentile_df,
        how="left",
        left_index=True, right_index=True,
    )

    merged_df.salary_percentile = merged_df.salary_percentile.fillna(0)
    
    return dict(
        salary_bins=bins,
        salary_percentile=pd.Series(merged_df.salary_percentile, dtype=int, index=clean_application_record.index),
    )


def years_employed(days_employed: pd.Series) -> pd.Series:
    """Convert days employed into days employed"""
    return days_employed.abs().div(365).astype(int)


def has_children(cnt_children: pd.Series) -> pd.Series:
    """Convert the number of children into a binary `has_children` yes/no"""
    return pd.Series(np.where(cnt_children>0, 1, 0), index=cnt_children.index)


def has_at_least_one_loan(
    clean_application_record: pd.DataFrame,
    status: pd.Series
) -> pd.Series:
    """Get the ID of users with at least one loan, i.e., not only X status
    NOTE be careful to not introduce data leakage in predictive task
    """
    user_ids_with_loan = status[status!="X"].index.unique()
    
    loan_mask = clean_application_record.index.isin(user_ids_with_loan)

    clean_application_record["has_at_least_one_loan"] = 0
    clean_application_record.loc[loan_mask, "has_at_least_one_loan"] = 1
    return clean_application_record.has_at_least_one_loan


def number_of_credit_months(clean_application_record: pd.DataFrame, month_index: pd.Series) -> pd.Series:
    """Total number of months of credit history per user"""
    user_max = month_index.groupby("id").max()
    merged_df = pd.merge(
        clean_application_record, user_max.to_frame(),
        how="left",
        left_index=True, right_index=True,
    )
    merged_df.month_index = merged_df.month_index.fillna(0)

    return pd.Series(merged_df.month_index, dtype=int, index=clean_application_record.index)


@save_to.parquet(path=value("./data/features.parquet"), output_name_="save_features")
def feature_set(
    eduction_type_encoded: pd.Series,
    family_status_encoded: pd.Series,
    housing_type_encoded: pd.Series,
    income_type_encoded: pd.Series,
    occupation_type_encoded: pd.Series,
    own_car_encoded: pd.Series,
    own_realty_encoded: pd.Series,
    number_of_credit_months: pd.Series,
    age_binned: pd.Series,
    income_percentile: pd.Series,
    has_children: pd.Series,
    salary_percentile: pd.Series,
    years_employed: pd.Series,
    cnt_fam_members: pd.Series,
) -> pd.DataFrame:
    """Join features into dataframe and save to parquet"""
    _df = pd.DataFrame.from_dict(locals())
    return _df


def days_due(status: pd.Series) -> pd.Series:
    """Encode status label into number of days as a regression target. 
    Original values indicate:
        X: No loan for the month
        C: paid off that month
        0: 1-29 days past due
        1: 30-59 days past due
        2: 60-89 days overdue
        3: 90-119 days overdue
        4: 120-149 days overdue
        5: Overdue or bad debts, write-offs for more than 150 days
    """
    encoding = {
        "X": -10,
        "C": 0,
        "0": 15,
        "1": 45,
        "2": 75,
        "3": 105,
        "4": 135,
        "5": 165,
    }
    return status.replace(encoding).astype(int)


@extract_fields(dict(
    days_due_mean=pd.Series,
    days_due_std=pd.Series,
))
def days_due_agg(clean_application_record: pd.DataFrame, days_due: pd.Series) -> dict:
    """Compute average and standard deviation of `days_due` per user"""
    aggregates = days_due.groupby("id").agg(["mean", "std"])

    return dict(
        days_due_mean=aggregates["mean"],
        days_due_std=aggregates["std"],
    )


@save_to.parquet(path=value("./data/targets.parquet"), output_name_="save_targets")
def target_set(
    days_due_mean: pd.Series,
    days_due_std: pd.Series,
) -> pd.DataFrame:
    """Join targets into dataframe and save to parquet"""
    _df = pd.DataFrame.from_dict(locals())
    return _df


@save_to.parquet(
    path=value("./data/preprocessed_dataset.parquet"), 
    output_name_="save_preprocessed_dataset",
)
def preprocessed_dataset(feature_set: pd.DataFrame, target_set: pd.DataFrame) -> pd.DataFrame:
    """Inner join between users `id` with features and targets available; save to parquet"""
    return pd.merge(feature_set, target_set, how="inner", left_index=True, right_index=True)
     