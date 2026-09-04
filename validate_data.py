import pandas as pd

df = pd.read_csv("Loan_default.csv")

required_columns = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
    "Default"
]

# Check columns
for column in required_columns:
    assert column in df.columns, f"Missing column: {column}"

# Check target values
assert set(df["Default"].unique()).issubset({0, 1})

# Check missing values
assert df.isnull().sum().sum() == 0

# Check important numeric columns
assert df["Age"].dtype != "object"
assert df["Income"].dtype != "object"
assert df["CreditScore"].dtype != "object"

print("Data validation PASSED!")