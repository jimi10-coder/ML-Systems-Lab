
from sklearn.datasets import fetch_openml
import os


def load_adult_dataset():
    """
    Load the raw UCI Adult dataset without modifying it.
    """
    print("Loading UCI Adult dataset...")

    dataset = fetch_openml(
        name="adult",
        version=2,
        as_frame=True
    )

    data = dataset.frame

    print("\nDataset loaded successfully!")
    print("Number of rows:", data.shape[0])
    print("Number of columns:", data.shape[1])

    return data



def validate_dataset(data):
    """Validate the Adult dataset without modifying or dropping rows."""

    print("\n========== DATA VALIDATION ==========")
    print("\nDataset shape:", data.shape)

    expected_columns = [
        "age", "workclass", "fnlwgt", "education",
        "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain",
        "capital-loss", "hours-per-week",
        "native-country", "class"
    ]

    missing_columns = set(expected_columns) - set(data.columns)
    extra_columns = set(data.columns) - set(expected_columns)

    print("\nMissing columns:", missing_columns)
    print("Extra columns:", extra_columns)

    # Numerical range validation
    range_rules = {
        "age": (17, 90),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356)
    }

    range_violations = {}

    print("\nRange checks:")

    for column, (minimum, maximum) in range_rules.items():
        invalid = (data[column] < minimum) | (data[column] > maximum)
        range_violations[column] = int(invalid.sum())
        print(column, "- violations:", range_violations[column])

    # Categorical value validation
    allowed_categories = {
        "workclass": [
            "Private", "Self-emp-not-inc", "Self-emp-inc",
            "Federal-gov", "Local-gov", "State-gov",
            "Without-pay", "Never-worked", "?"
        ],
        "education": [
            "Bachelors", "Some-college", "11th", "HS-grad",
            "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
            "7th-8th", "12th", "Masters", "1st-4th",
            "10th", "Doctorate", "5th-6th", "Preschool"
        ],
        "marital-status": [
            "Married-civ-spouse", "Divorced", "Never-married",
            "Separated", "Widowed", "Married-spouse-absent",
            "Married-AF-spouse"
        ],
        "occupation": [
            "Tech-support", "Craft-repair", "Other-service",
            "Sales", "Exec-managerial", "Prof-specialty",
            "Handlers-cleaners", "Machine-op-inspct",
            "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv",
            "Protective-serv", "Armed-Forces", "?"
        ],
        "relationship": [
            "Wife", "Own-child", "Husband", "Not-in-family",
            "Other-relative", "Unmarried"
        ],
        "race": [
            "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
            "Other", "Black"
        ],
        "sex": ["Female", "Male"],
        "native-country": [
            "United-States", "Cambodia", "England", "Puerto-Rico",
            "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
            "India", "Japan", "Greece", "South", "China", "Cuba",
            "Iran", "Honduras", "Philippines", "Italy", "Poland",
            "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
            "France", "Dominican-Republic", "Laos", "Ecuador",
            "Taiwan", "Haiti", "Columbia", "Hungary",
            "Guatemala", "Nicaragua", "Scotland", "Thailand",
            "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
            "Peru", "Hong", "Holand-Netherlands", "?"
        ],
        "class": ["<=50K", ">50K"]
    }

    categorical_violations = {}

    print("\nCategorical value checks:")

    for column, allowed_values in allowed_categories.items():
        values = data[column].astype("string").str.strip()
        invalid = values.notna() & ~values.isin(allowed_values)
        categorical_violations[column] = int(invalid.sum())
        print(column, "- invalid values:", categorical_violations[column])

    # Missing-value validation
    print("\nMissing values:")

    missing_report = {}

    for column in data.columns:
        actual_missing = int(data[column].isna().sum())
        question_marks = int(
            (data[column].astype("string").str.strip() == "?").sum()
        )
        total_missing = actual_missing + question_marks
        missing_report[column] = total_missing

        if total_missing > 0:
            print(column, ":", total_missing)

    # Display data types
    print("\nData types:")
    print(data.dtypes)

    # Save validation report
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    report_path = os.path.join(results_dir, "validation_report.txt")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("UCI ADULT DATASET - VALIDATION REPORT\n")
        file.write("=" * 45 + "\n")
        file.write(f"Rows: {data.shape[0]}\n")
        file.write(f"Columns: {data.shape[1]}\n")
        file.write(f"Missing columns: {len(missing_columns)}\n")
        file.write(f"Extra columns: {len(extra_columns)}\n\n")

        file.write("Range violations:\n")
        for column, count in range_violations.items():
            file.write(f"{column}: {count}\n")

        file.write("\nCategorical value violations:\n")
        for column, count in categorical_violations.items():
            file.write(f"{column}: {count}\n")

        file.write("\nMissing values:\n")
        for column, count in missing_report.items():
            file.write(f"{column}: {count}\n")

        file.write("\nData types:\n")
        file.write(data.dtypes.to_string())

    print("\nValidation report saved:", report_path)
    print("\n========== VALIDATION COMPLETE ==========")



if __name__ == "__main__":
    df = load_adult_dataset()

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    validate_dataset(df)
