import pandas as pd

from hamilton.function_modifiers import extract_columns, load_from, value


CLEAN_APPLICATION_RECORD_COLUMNS = [
    'cnt_children', 'amt_income_total', 'days_birth', 'days_employed',
    'flag_mobil', 'flag_work_phone', 'flag_phone', 'flag_email',
    'cnt_fam_members', 'eduction_type_encoded', 'family_status_encoded',
    'gender_encoded', 'housing_type_encoded', 'income_type_encoded',
    'occupation_type_encoded', 'own_car_encoded', 'own_realty_encoded'
]

@extract_columns(*CLEAN_APPLICATION_RECORD_COLUMNS)
@load_from.parquet(path=value("./data/application_clean.parquet"))
def clean_application_record(data: pd.DataFrame) -> pd.DataFrame:
    return data


def has_children(cnt_children: pd.Series) -> pd.Series:
    return pd.Series(np.where(cnt_children>0, 1, 0), index=cnt_children.index)


def age_in_years(days_birth: pd.Series) -> pd.Series:
    return days_birth.abs().div(365).astype(int)

