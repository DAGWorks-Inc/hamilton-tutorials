import numpy as np
import pandas as pd

from hamilton.function_modifiers import extract_columns, load_from, value, save_to, does, value


def _sanitize_column_names(column_names: list[str]) -> list[str]:
    return [col.lower() for col in column_names]


RAW_APPLICATION_RECORD_COLUMNS = [
    'code_gender', 'flag_own_car', 'flag_own_realty',
    'cnt_children', 'amt_income_total', 'name_income_type',
    'name_education_type', 'name_family_status', 'name_housing_type',
    'days_birth', 'days_employed', 'flag_mobil', 'flag_work_phone',
    'flag_phone', 'flag_email', 'occupation_type', 'cnt_fam_members'
]

@extract_columns(*RAW_APPLICATION_RECORD_COLUMNS)
@load_from.csv(path=value("./data/application_record.csv"))
def raw_application_record(data: pd.DataFrame) -> pd.DataFrame:
    data = data.set_index("ID")
    data.columns = _sanitize_column_names(data.columns)
    return data


RAW_CREDIT_RECORD_COLUMNS = ['months_balance', 'status']

@extract_columns(*RAW_CREDIT_RECORD_COLUMNS)
@load_from.csv(path=value("./data/credit_record.csv"))
def raw_credit_record(data: pd.DataFrame) -> pd.DataFrame:
    data = data.set_index("ID")
    data.columns = _sanitize_column_names(data.columns)
    return data


def gender_encoded(code_gender: pd.Series) -> pd.Series:
    encoding = {"M": 0, "F": 1}
    return code_gender.replace(encoding)


def own_car_encoded(flag_own_car: pd.Series) -> pd.Series:
    encoding = {"N": 0, "Y": 1}
    return flag_own_car.replace(encoding)


def own_realty_encoded(flag_own_realty: pd.Series) -> pd.Series:
    encoding = {"N": 0, "Y": 1}
    return flag_own_realty.replace(encoding)


def eduction_type_encoded(name_education_type: pd.Series) -> pd.Series:
    encoding = {
        'Lower secondary': 0,
        'Secondary / secondary special': 1,
        'Incomplete higher': 2,
        'Higher education': 3,
        'Academic degree': 4,
    }
    return name_education_type.replace(encoding)


def income_type_encoded(name_income_type: pd.Series) -> pd.Series:
    encoding = {
        'Student': 0,
        'Working': 1,
        'Commercial associate': 2,
        'Pensioner': 3,
        'State servant': 4,
    }
    return name_income_type.replace(encoding)


def family_status_encoded(name_family_status: pd.Series) -> pd.Series:
    encoding = {
        'Single / not married': 0,
        'Married': 1,
        'Civil marriage': 2,
        'Separated': 3,
        'Widow': 4,
    }
    return name_family_status.replace(encoding)


def housing_type_encoded(name_housing_type: pd.Series) -> pd.Series:
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


def _join_series(**series) -> pd.DataFrame:
    return pd.DataFrame(dict(**series))


@save_to.parquet(path=value("./data/application_clean.parquet"), output_name_="save_application_clean")
@does(_join_series)
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
    pass


