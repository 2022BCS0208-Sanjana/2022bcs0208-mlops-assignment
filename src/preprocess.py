import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Clean spaces
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Drop useless column
    df = df.drop(columns=["loan_id"])

    # Encode categorical
    df['education'] = df['education'].map({
        'Graduate': 1,
        'Not Graduate': 0
    })

    df['self_employed'] = df['self_employed'].map({
        'Yes': 1,
        'No': 0
    })

    df['loan_status'] = df['loan_status'].map({
        'Approved': 1,
        'Rejected': 0
    })

    return df
