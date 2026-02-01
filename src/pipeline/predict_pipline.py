# import  numpy as np

if __name__ == "__main__":
    report = {
        "LinearRegression": {
            "train_score": 0.72,
            "test_score": 0.70,
            "gap": 0.02
        },
        "Ridge": {
            "train_score": 0.78,
            "test_score": 0.76,
            "gap": 0.02
        },
        "RandomForest": {
            "train_score": 0.98,
            "test_score": 0.74,
            "gap": 0.24
        },
        "XGBoost": {
            "train_score": 0.92,
            "test_score": 0.80,
            "gap": 0.12
        }
    }

    print(report.keys())
    print(list(report.values()))

    model_report = {
    "LinearRegression": 0.71,
    "Ridge": 0.75,
    "RandomForest": 0.69,
    "XGBoost": 0.81
    }

    # print(max(model_report.values())

    print(list(model_report.keys())[
        list(model_report.values()).index(max(model_report.values()))
    ])
