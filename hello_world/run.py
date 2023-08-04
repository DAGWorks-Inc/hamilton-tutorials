import importlib

import pandas as pd
from hamilton import driver


def main() -> None:
    """Simple """

    # importing your function modules from within `main()` help
    # indicate they are meant to be used in the Hamilton Driver
    import functions
    functions2 = importlib.import_module("functions2")

    input_data = {
        "signups": pd.Series([1, 10, 50, 100, 200, 400]),
        "spend": pd.Series([10, 10, 20, 40, 40, 50]),
    }

    initial_config = {}
    dr = driver.Driver(initial_config, functions, functions2)

    output_columns = [
        "spend",
        "signups",
        "avg_3wk_spend",
        "spend_per_signup",
        "spend_zero_mean_unit_variance",
    ]

    df = dr.execute(
        output_columns,
        inputs=input_data
    )
    print(df.head())


if __name__ == "__main__":
    main()
