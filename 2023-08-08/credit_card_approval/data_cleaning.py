import numpy as np
import pandas as pd

from hamilton.function_modifiers import (
    extract_columns,
    load_from,
    value,
    save_to,
)


RAW_APPLICATION_RECORD_COLUMNS = [
    'code_gender', 'flag_own_car', 'flag_own_realty',
    'cnt_children', 'amt_income_total', 'name_income_type',
    'name_education_type', 'name_family_status', 'name_housing_type',
    'days_birth', 'days_employed', 'flag_mobil', 'flag_work_phone',
    'flag_phone', 'flag_email', 'occupation_type', 'cnt_fam_members'
]

@extract_columns(*RAW_APPLICATION_RECORD_COLUMNS)
@load_from.csv(path=value("./data/application_record.csv"))
def raw_application_record(raw_application_csv: pd.DataFrame) -> pd.DataFrame:
    """Load the raw application data from CSV; rename columns to lowercase;  set ID as index"""
    raw_application_csv.columns = [col.lower() for col in raw_application_csv.columns]
    raw_application_csv = raw_application_csv.set_index("id")
    raw_application_csv = raw_application_csv.groupby("id").first()
    return raw_application_csv


def gender_encoded(code_gender: pd.Series) -> pd.Series:
    """Encode code_gender as int"""
    encoding = {"M": 0, "F": 1}
    return code_gender.replace(encoding)


def own_car_encoded(flag_own_car: pd.Series) -> pd.Series:
    """encode flag_own_car as int"""
    encoding = {"N": 0, "Y": 1}
    return flag_own_car.replace(encoding)


def own_realty_encoded(flag_own_realty: pd.Series) -> pd.Series:
    """encode flag_own_realty as int"""
    encoding = {"N": 0, "Y": 1}
    return flag_own_realty.replace(encoding)


def eduction_type_encoded(name_education_type: pd.Series) -> pd.Series:
    """encode name_education_type as int"""
    encoding = {
        'Lower secondary': 0,
        'Secondary / secondary special': 1,
        'Incomplete higher': 2,
        'Higher education': 3,
        'Academic degree': 4,
    }
    return name_education_type.replace(encoding)


def income_type_encoded(name_income_type: pd.Series) -> pd.Series:
    """encode name_income_type as int"""
    encoding = {
        'Student': 0,
        'Working': 1,
        'Commercial associate': 2,
        'Pensioner': 3,
        'State servant': 4,
    }
    return name_income_type.replace(encoding)


def family_status_encoded(name_family_status: pd.Series) -> pd.Series:
    """encode name_family_status as int"""
    encoding = {
        'Single / not married': 0,
        'Married': 1,
        'Civil marriage': 2,
        'Separated': 3,
        'Widow': 4,
    }
    return name_family_status.replace(encoding)


def housing_type_encoded(name_housing_type: pd.Series) -> pd.Series:
    """encode name_housing_type as int"""
    encoding = {
        'With parents': 0,
        'Rented apartment': 1,
        'Co-op apartment': 2,
        'Municipal apartment': 3,
        'House / apartment': 4,
        'Office apartment': 5,
    }
    return name_housing_type.replace(encoding)


def occupation_type_encoded(occupation_type: pd.Series) -> pd.Series:
    """encode occupation_type as int"""
    encoding = {
        np.nan: 0,  # np.nan indicates unemployed
        'Laborers': 1,
        'Core staff': 2,
        'Sales staff': 3,
        'Managers': 4,
        'Drivers': 5,
        'High skill tech staff': 6,
        'Accountants': 7,
        'Medicine staff': 8,
        'Cooking staff': 9,
        'Security staff': 10,
        'Cleaning staff': 11,
        'Private service staff': 12,
        'Low-skill Laborers': 13,
        'Secretaries': 14,
        'Waiters/barmen staff': 15,
        'Realty agents': 16,
        'HR staff': 17,
        'IT staff': 18,
    }
    return occupation_type.replace(encoding)


@save_to.parquet(path=value("./data/application_clean.parquet"), output_name_="save_application_clean")
def application_clean(
    cnt_children: pd.Series,
    amt_income_total: pd.Series,
    days_birth: pd.Series,
    days_employed: pd.Series,
    flag_mobil: pd.Series,
    flag_work_phone: pd.Series,
    flag_phone: pd.Series,
    flag_email: pd.Series,
    cnt_fam_members: pd.Series,
    eduction_type_encoded: pd.Series,
    family_status_encoded: pd.Series,
    gender_encoded: pd.Series,
    housing_type_encoded: pd.Series,
    income_type_encoded: pd.Series,
    occupation_type_encoded: pd.Series,
    own_car_encoded: pd.Series,
    own_realty_encoded: pd.Series,
) -> pd.DataFrame:
    """Join cleaned series into dataframe and save to parquet"""
    _df = pd.DataFrame.from_dict(locals())
    return _df


RAW_CREDIT_RECORD_COLUMNS = ['months_balance', 'status']

@extract_columns(*RAW_CREDIT_RECORD_COLUMNS)
@load_from.csv(path=value("./data/credit_record.csv"))
def raw_credit_record(raw_credit_csv: pd.DataFrame) -> pd.DataFrame:
    """Load the raw credit data from CSV; rename columns to lowercase;  set ID as index"""
    raw_credit_csv.columns = [col.lower() for col in raw_credit_csv.columns]
    raw_credit_csv = raw_credit_csv.set_index("id")
    return raw_credit_csv


def month_index(raw_credit_record: pd.DataFrame) -> pd.Series:
    """Convert `months_balance` from -3, -2, 1, 0 to 0, 1, 2, 3"""
    return (
        raw_credit_record["months_balance"]
        .groupby("id")
        .apply(lambda x: x - x.min())
        .reset_index(level=0, drop=True)
    )


@save_to.parquet(path=value("./data/credit_clean.parquet"), output_name_="save_credit_clean")
def credit_clean(
    month_index: pd.Series,
    status: pd.Series,
) -> pd.DataFrame:
    """Join cleaned series into dataframe and save to parquet"""
    _df = pd.DataFrame.from_dict(locals())
    return _df