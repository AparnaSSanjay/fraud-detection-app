"""Fraud scoring pipeline aligned with Fraud_Detection.ipynb.

The notebook's prediction path is post-transaction (scenario A): new balances are
known. `step` is excluded because, in this file, every legitimate row is in
steps 1–7 while fraud extends to step 95, so the column leaks the label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
INVESTIGATION_COST = 25.0
FEE = 0.01
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
RAW_NUMERIC = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
FEATURE_COLUMNS = [
    "log_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origChange",
    "destChange",
    "errOrig",
    "errDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "log_oldbalanceOrg",
    "log_newbalanceOrig",
    "log_oldbalanceDest",
    "log_newbalanceDest",
    "orig_emptied",
    "amount_eq_oldbal",
    "dest_zero",
    "orig_recon_ok",
    "dest_recon_ok",
    "risky_type",
    "emptied_risky",
]
FEATURE_LABELS = {
    "log_amount": "Log amount",
    "oldbalanceOrg": "Sender balance before",
    "newbalanceOrig": "Sender balance after",
    "oldbalanceDest": "Recipient balance before",
    "newbalanceDest": "Recipient balance after",
    "origChange": "Sender balance change",
    "destChange": "Recipient balance change",
    "errOrig": "Sender ledger error",
    "errDest": "Recipient ledger error",
    "type_CASH_IN": "Type is CASH_IN",
    "type_CASH_OUT": "Type is CASH_OUT",
    "type_DEBIT": "Type is DEBIT",
    "type_PAYMENT": "Type is PAYMENT",
    "type_TRANSFER": "Type is TRANSFER",
    "log_oldbalanceOrg": "Log sender balance before",
    "log_newbalanceOrig": "Log sender balance after",
    "log_oldbalanceDest": "Log recipient balance before",
    "log_newbalanceDest": "Log recipient balance after",
    "orig_emptied": "Sender account emptied",
    "amount_eq_oldbal": "Amount equals sender balance",
    "dest_zero": "Recipient balance stays zero",
    "orig_recon_ok": "Sender ledger reconciles",
    "dest_recon_ok": "Recipient ledger reconciles",
    "risky_type": "TRANSFER or CASH_OUT",
    "emptied_risky": "Emptied and risky type",
}
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
]
# Same ranges as notebook section 4b. The search scores PR-AUC with 3-fold CV
# on the training split only. Validation and test are not used to pick parameters.
SEARCH_DRAWS = 12
SEARCH_NOTES = {
    "Logistic Regression": "Grid over C in 0.01, 0.1, 1, 10 and 100. Class weights stay balanced. A scaler is fit inside each training fold.",
    "Random Forest": "12 draws over trees, depth, leaf size and feature fraction. Class weights stay balanced.",
    "XGBoost": "12 draws over trees, depth, learning rate, row and column subsamples, child weight and L2. The fraud weight is fixed from the training split.",
    "LightGBM": "12 draws over trees, leaves, depth, learning rate, subsamples, child samples and L2. The fraud weight is fixed from the training split.",
}
FEATURE_HELP = {
    "log_amount": "Log of one plus the amount, because amounts are heavily right-skewed.",
    "oldbalanceOrg": "Sender balance before the transaction.",
    "newbalanceOrig": "Sender balance after the transaction. This is post-transaction information.",
    "oldbalanceDest": "Recipient balance before the transaction.",
    "newbalanceDest": "Recipient balance after the transaction.",
    "origChange": "How much the sender balance moved.",
    "destChange": "How much the recipient balance moved.",
    "errOrig": "Sender balance change minus the amount. Near zero means the ledger matches.",
    "errDest": "Recipient balance change versus the amount.",
    "type_CASH_IN": "1 when the type is CASH_IN.",
    "type_CASH_OUT": "1 when the type is CASH_OUT.",
    "type_DEBIT": "1 when the type is DEBIT.",
    "type_PAYMENT": "1 when the type is PAYMENT.",
    "type_TRANSFER": "1 when the type is TRANSFER.",
    "log_oldbalanceOrg": "Log of the sender balance before.",
    "log_newbalanceOrig": "Log of the sender balance after.",
    "log_oldbalanceDest": "Log of the recipient balance before.",
    "log_newbalanceDest": "Log of the recipient balance after.",
    "orig_emptied": "1 when the sender balance falls from a positive amount to exactly zero.",
    "amount_eq_oldbal": "1 when the amount equals the sender's old balance. This is the dominant fraud pattern in the file.",
    "dest_zero": "1 when the recipient balance is zero before and after.",
    "orig_recon_ok": "1 when the sender ledger matches the amount within 1 unit.",
    "dest_recon_ok": "1 when the recipient ledger matches the amount within 1 unit.",
    "risky_type": "1 for TRANSFER or CASH_OUT, the only types that contain fraud here.",
    "emptied_risky": "1 when a risky type also empties the sender account.",
}
FEATURE_GROUPS = [
    ("Amount and balances", ["log_amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "log_oldbalanceOrg", "log_newbalanceOrig", "log_oldbalanceDest", "log_newbalanceDest"]),
    ("Ledger", ["origChange", "destChange", "errOrig", "errDest", "orig_recon_ok", "dest_recon_ok"]),
    ("Type", ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER", "risky_type"]),
    ("Patterns", ["orig_emptied", "amount_eq_oldbal", "dest_zero", "emptied_risky"]),
]
HAND_CASES = [
    {
        "type": "TRANSFER",
        "amount": 500000,
        "oldbalanceOrg": 500000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "expected": "FRAUD",
        "note": "Drains the whole sender balance",
    },
    {
        "type": "CASH_OUT",
        "amount": 1200000,
        "oldbalanceOrg": 1200000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "expected": "FRAUD",
        "note": "Large cash-out of the full balance",
    },
    {
        "type": "TRANSFER",
        "amount": 90000,
        "oldbalanceOrg": 90000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 5000,
        "newbalanceDest": 95000,
        "expected": "FRAUD",
        "note": "Amount equals the old balance",
    },
    {
        "type": "PAYMENT",
        "amount": 120,
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 4880,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "expected": "LEGIT",
        "note": "Small payment, balance remains",
    },
    {
        "type": "CASH_IN",
        "amount": 8000,
        "oldbalanceOrg": 20000,
        "newbalanceOrig": 28000,
        "oldbalanceDest": 50000,
        "newbalanceDest": 42000,
        "expected": "LEGIT",
        "note": "Cash-in increases the sender balance",
    },
    {
        "type": "TRANSFER",
        "amount": 3000,
        "oldbalanceOrg": 90000,
        "newbalanceOrig": 87000,
        "oldbalanceDest": 10000,
        "newbalanceDest": 13000,
        "expected": "LEGIT",
        "note": "Transfer that does not empty the account",
    },
    {
        "type": "DEBIT",
        "amount": 400,
        "oldbalanceOrg": 6000,
        "newbalanceOrig": 5600,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "expected": "LEGIT",
        "note": "Small debit",
    },
    {
        "type": "CASH_OUT",
        "amount": 2500,
        "oldbalanceOrg": 80000,
        "newbalanceOrig": 77500,
        "oldbalanceDest": 20000,
        "newbalanceDest": 17500,
        "expected": "LEGIT",
        "note": "Cash-out of a small fraction of the balance",
    },
]


class DatasetError(Exception):
    pass


class ModelNotReady(Exception):
    pass


def find_dataset(start: Path | None = None) -> Path:
    root = start or Path(__file__).resolve().parent
    candidates = [
        root / "dataset" / "Fraud_Analysis_Dataset.xlsx",
        root / "Fraud_Analysis_Dataset.xlsx",
        root.parent / "dataset" / "Fraud_Analysis_Dataset.xlsx",
        root.parent.parent / "dataset" / "Fraud_Analysis_Dataset.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise DatasetError(
        "Fraud_Analysis_Dataset.xlsx was not found next to the app or in a dataset/ folder."
    )


def load_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, engine="openpyxl")
    required = ["step", "type", "isFraud", *RAW_NUMERIC]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetError(f"Dataset is missing columns: {', '.join(missing)}")
    unexpected = sorted(set(frame["type"].astype(str)) - set(TYPES))
    if unexpected:
        raise DatasetError(f"Unexpected transaction types: {', '.join(unexpected)}")
    if not frame.index.is_unique:
        raise DatasetError("Dataset index is not unique.")
    frame = frame.copy()
    frame["type"] = frame["type"].astype(str)
    frame["isFraud"] = frame["isFraud"].astype(int)
    for column in RAW_NUMERIC + ["step"]:
        frame[column] = pd.to_numeric(frame[column])
    return frame


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Same columns and formulas as `features()` in the notebook, without `step`."""
    amount = raw["amount"].astype(float)
    old_orig = raw["oldbalanceOrg"].astype(float)
    new_orig = raw["newbalanceOrig"].astype(float)
    old_dest = raw["oldbalanceDest"].astype(float)
    new_dest = raw["newbalanceDest"].astype(float)
    frame = pd.DataFrame(
        {
            "log_amount": np.log1p(amount),
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "origChange": old_orig - new_orig,
            "destChange": new_dest - old_dest,
            "errOrig": old_orig - new_orig - amount,
            "errDest": old_dest + amount - new_dest,
        },
        index=raw.index,
    )
    type_values = raw["type"].astype(str)
    for name in TYPES:
        frame[f"type_{name}"] = (type_values == name).astype(int)
    for column in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        frame[f"log_{column}"] = np.log1p(raw[column].astype(float))
    frame["orig_emptied"] = ((new_orig == 0) & (old_orig > 0)).astype(int)
    frame["amount_eq_oldbal"] = (amount == old_orig).astype(int)
    frame["dest_zero"] = ((old_dest == 0) & (new_dest == 0)).astype(int)
    frame["orig_recon_ok"] = (frame["errOrig"].abs() < 1).astype(int)
    frame["dest_recon_ok"] = (frame["errDest"].abs() < 1).astype(int)
    frame["risky_type"] = (
        (frame["type_TRANSFER"] == 1) | (frame["type_CASH_OUT"] == 1)
    ).astype(int)
    frame["emptied_risky"] = (frame["orig_emptied"] * frame["risky_type"]).astype(int)
    frame = frame.loc[:, FEATURE_COLUMNS]
    if frame.isna().any().any():
        raise DatasetError("Feature building produced missing values.")
    return frame


def _json_float(value: float, digits: int = 6) -> float:
    return float(np.round(float(value), digits))


def dataset_summary(raw: pd.DataFrame, y: pd.Series, splits: dict[str, pd.Series]) -> dict[str, Any]:
    fraud = y == 1
    legit = ~fraud
    by_type = []
    for name in TYPES:
        subset = raw["type"] == name
        count = int(subset.sum())
        fraud_count = int((subset & fraud).sum())
        by_type.append(
            {
                "type": name,
                "count": count,
                "fraud": fraud_count,
                "legit": count - fraud_count,
                "fraud_rate": (fraud_count / count) if count else 0.0,
            }
        )
    log_amount = np.log1p(raw["amount"].astype(float))
    edges = np.linspace(float(log_amount.min()), float(log_amount.max()), 21)
    legit_hist, _ = np.histogram(log_amount[legit], bins=edges)
    fraud_hist, _ = np.histogram(log_amount[fraud], bins=edges)
    amount_labels = [
        f"{np.expm1(edges[i]):,.0f}–{np.expm1(edges[i + 1]):,.0f}"
        for i in range(len(edges) - 1)
    ]
    step_max = int(raw["step"].max())
    step_edges = np.arange(1, step_max + 2)
    legit_steps, _ = np.histogram(raw.loc[legit, "step"], bins=step_edges)
    fraud_steps, _ = np.histogram(raw.loc[fraud, "step"], bins=step_edges)
    legit_step_max = int(raw.loc[legit, "step"].max())
    leaked = int((fraud & (raw["step"] > legit_step_max)).sum())
    err_orig = raw["oldbalanceOrg"] - raw["newbalanceOrig"] - raw["amount"]
    err_dest = raw["oldbalanceDest"] + raw["amount"] - raw["newbalanceDest"]
    fraud_median = float(raw.loc[fraud, "amount"].median())
    legit_median = float(raw.loc[legit, "amount"].median())
    emptied = int(((raw["newbalanceOrig"] == 0) & fraud).sum())
    transfer_fraud = int(((raw["type"] == "TRANSFER") & fraud).sum())
    cashout_fraud = int(((raw["type"] == "CASH_OUT") & fraud).sum())
    split_counts = {}
    for name, labels in splits.items():
        split_counts[name] = {"rows": int(len(labels)), "fraud": int(labels.sum())}
    findings = [
        f"{len(raw):,} transactions, {int(fraud.sum()):,} fraud ({float(y.mean()):.2%}). "
        f"Missing values: {int(raw.isna().sum().sum())}. Duplicates: {int(raw.duplicated().sum())}.",
        f"Fraud occurs only in TRANSFER ({transfer_fraud:,}) and CASH_OUT ({cashout_fraud:,}). "
        f"PAYMENT, CASH_IN and DEBIT have none in this file.",
        f"Median fraud amount is {fraud_median:,.0f} versus {legit_median:,.0f} for legitimate "
        f"transactions ({(fraud_median / legit_median) if legit_median else 0:.0f}×).",
        f"{emptied:,} of {int(fraud.sum()):,} frauds leave the sender balance at exactly 0.",
        f"Median sender ledger error is {float(err_orig[fraud].median()):,.0f} for fraud and "
        f"{float(err_orig[legit].median()):,.0f} for legitimate transactions. "
        f"Median recipient ledger error is {float(err_dest[fraud].median()):,.0f} for fraud.",
        f"step leaks the label: legitimate rows stop at step {legit_step_max}, and {leaked:,} fraud rows "
        f"are later. step is not used as a feature.",
        "This is a fraud-enriched PaySim sample (about 10% fraud). The rates are not the rates of a real book of business.",
    ]
    examples = _example_rows(raw)
    return {
        "rows": int(len(raw)),
        "fraud": int(fraud.sum()),
        "legit": int(legit.sum()),
        "fraud_rate": float(y.mean()),
        "missing": int(raw.isna().sum().sum()),
        "duplicates": int(raw.duplicated().sum()),
        "median_amount_fraud": fraud_median,
        "median_amount_legit": legit_median,
        "emptied_frauds": emptied,
        "legit_max_step": legit_step_max,
        "fraud_rows_after_legit_step": leaked,
        "by_type": by_type,
        "amount_hist": {
            "labels": amount_labels,
            "legit": [int(v) for v in legit_hist],
            "fraud": [int(v) for v in fraud_hist],
        },
        "step_hist": {
            "labels": [str(int(step)) for step in step_edges[:-1]],
            "legit": [int(v) for v in legit_steps],
            "fraud": [int(v) for v in fraud_steps],
        },
        "split": split_counts,
        "findings": findings,
        "examples": examples,
        "hand_cases": HAND_CASES,
    }


def _example_rows(raw: pd.DataFrame) -> list[dict[str, Any]]:
    specs = [
        ("Fraud TRANSFER from the file", (raw["isFraud"] == 1) & (raw["type"] == "TRANSFER")),
        ("Fraud CASH_OUT from the file", (raw["isFraud"] == 1) & (raw["type"] == "CASH_OUT")),
        ("Legitimate PAYMENT", (raw["isFraud"] == 0) & (raw["type"] == "PAYMENT")),
        ("Legitimate CASH_IN", (raw["isFraud"] == 0) & (raw["type"] == "CASH_IN")),
        ("Legitimate DEBIT", (raw["isFraud"] == 0) & (raw["type"] == "DEBIT")),
        ("Legitimate TRANSFER", (raw["isFraud"] == 0) & (raw["type"] == "TRANSFER")),
        ("Legitimate CASH_OUT", (raw["isFraud"] == 0) & (raw["type"] == "CASH_OUT")),
    ]
    examples = []
    for note, mask in specs:
        subset = raw.loc[mask]
        if subset.empty:
            continue
        row = subset.iloc[0]
        examples.append(
            {
                "id": int(row.name),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "expected": "FRAUD" if int(row["isFraud"]) == 1 else "LEGIT",
                "note": note,
            }
        )
    return examples


@dataclass
class Bundle:
    path: Path
    raw: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    amount: pd.Series
    Xtr: pd.DataFrame
    Xva: pd.DataFrame
    Xte: pd.DataFrame
    ytr: pd.Series
    yva: pd.Series
    yte: pd.Series
    summary: dict[str, Any] = field(repr=False)

    @property
    def positive_weight(self) -> float:
        return float((self.ytr == 0).sum()) / float((self.ytr == 1).sum())


def make_bundle(path: Path | None = None) -> Bundle:
    from sklearn.model_selection import train_test_split

    dataset_path = path or find_dataset()
    raw = load_raw(dataset_path)
    X = build_features(raw)
    y = raw["isFraud"].astype(int)
    amount = raw["amount"].astype(float)
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=SEED
    )
    summary = dataset_summary(raw, y, {"train": ytr, "validation": yva, "test": yte})
    summary["path"] = str(dataset_path)
    summary["scenario"] = "post-transaction"
    summary["excluded_features"] = ["step", "nameOrig", "nameDest"]
    return Bundle(
        path=dataset_path,
        raw=raw,
        X=X,
        y=y,
        amount=amount,
        Xtr=Xtr,
        Xva=Xva,
        Xte=Xte,
        ytr=ytr,
        yva=yva,
        yte=yte,
        summary=summary,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _search(name: str, positive_weight: float, n_jobs: int = -1):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_NAMES)}.")
    folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
    if name == "Logistic Regression":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
        )
        return GridSearchCV(
            estimator,
            {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
            cv=folds,
            scoring="average_precision",
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
    if name == "Random Forest":
        estimator = RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=SEED)
        space = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8, 12],
            "min_samples_leaf": [2, 5, 10, 20],
            "max_features": ["sqrt", 0.5],
        }
    elif name == "XGBoost":
        estimator = XGBClassifier(
            scale_pos_weight=positive_weight,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
        space = {
            "n_estimators": [150, 300, 500],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1, 5, 10],
        }
    else:
        estimator = LGBMClassifier(
            scale_pos_weight=positive_weight,
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            subsample_freq=1,
        )
        space = {
            "n_estimators": [150, 300, 500],
            "num_leaves": [7, 15, 31],
            "max_depth": [-1, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [10, 20, 40],
            "reg_lambda": [1, 5, 10],
        }
    return RandomizedSearchCV(
        estimator,
        space,
        n_iter=SEARCH_DRAWS,
        cv=folds,
        scoring="average_precision",
        random_state=SEED,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )


def parameter_rows(params: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in params.items():
        rows.append({"name": str(key).split("__")[-1], "value": value})
    return rows


def classification_report(y_true, proba, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def decision_cost(y_true, proba, amounts, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    predicted = proba >= threshold
    missed = (~predicted) & (y_true == 1)
    false_alarms = predicted & (y_true == 0)
    return float(
        amounts[missed].sum()
        + predicted.sum() * INVESTIGATION_COST
        + FEE * amounts[false_alarms].sum()
    )


def choose_thresholds(y_true, proba, amounts) -> tuple[float, float]:
    grid = np.logspace(-4, np.log10(0.99), 200)
    f1_scores = [f1_score(y_true, proba >= threshold, zero_division=0) for threshold in grid]
    costs = [decision_cost(y_true, proba, amounts, threshold) for threshold in grid]
    return float(grid[int(np.argmax(f1_scores))]), float(grid[int(np.argmin(costs))])


def feature_importance(name: str, estimator, columns: list[str]) -> list[dict[str, Any]]:
    if name == "Logistic Regression":
        coefficients = np.abs(estimator.named_steps["logisticregression"].coef_[0])
        total = float(coefficients.sum()) or 1.0
        values = coefficients / total
        kind = "absolute coefficient share"
    else:
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "gain share"
    order = np.argsort(values)[::-1]
    rows = []
    for index in order[:12]:
        feature = columns[int(index)]
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "importance": _json_float(values[int(index)]),
                "kind": kind,
            }
        )
    return rows


def financials(y_true, proba, amounts, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    predicted = proba >= threshold
    fraud = y_true == 1
    legit = ~fraud
    missed = amounts[(~predicted) & fraud].sum()
    fraud_value = amounts[fraud].sum()
    review_cost = float(predicted.sum() * INVESTIGATION_COST)
    lost_fee = float(FEE * amounts[predicted & legit].sum())
    revenue = float(FEE * amounts[(~predicted) & legit].sum())
    profit = float(revenue - missed - review_cost)
    no_model_profit = float(FEE * amounts[legit].sum() - fraud_value)
    return {
        "fraud_value": float(fraud_value),
        "missed_loss": float(missed),
        "review_cost": review_cost,
        "lost_fee": lost_fee,
        "revenue": revenue,
        "profit": profit,
        "no_model_profit": no_model_profit,
        "uplift": float(profit - no_model_profit),
        "fraud_prevented": float(amounts[predicted & fraud].sum()),
    }


def _curve_points(x_values, y_values, limit: int = 80) -> list[dict[str, float]]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if len(x_values) > limit:
        indexes = np.linspace(0, len(x_values) - 1, limit).astype(int)
        x_values = x_values[indexes]
        y_values = y_values[indexes]
    return [
        {"x": _json_float(x, 5), "y": _json_float(y, 5)}
        for x, y in zip(x_values, y_values)
    ]


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    medians: dict[str, float]
    threshold_f1: float
    threshold_cost: float
    applied_threshold: float
    trained_at: str
    positive_weight: float
    validation_at_half: dict[str, float]
    test_at_f1: dict[str, float]
    overfit: dict[str, float]
    importance: list[dict[str, Any]]
    best_params: dict[str, Any]
    cv_score: float
    insights: list[str]
    train_proba: np.ndarray
    val_proba: np.ndarray
    test_proba: np.ndarray

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "estimator": self.estimator,
            "feature_columns": FEATURE_COLUMNS,
            "medians": self.medians,
            "threshold_f1": self.threshold_f1,
            "threshold_cost": self.threshold_cost,
            "applied_threshold": self.applied_threshold,
            "trained_at": self.trained_at,
            "positive_weight": self.positive_weight,
            "validation_at_half": self.validation_at_half,
            "test_at_f1": self.test_at_f1,
            "overfit": self.overfit,
            "importance": self.importance,
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "insights": self.insights,
            "scenario": "post-transaction",
        }

    @classmethod
    def from_artifact(cls, payload: dict[str, Any], bundle: Bundle) -> "TrainedModel":
        columns = payload.get("feature_columns")
        if columns != FEATURE_COLUMNS:
            raise ModelNotReady(
                "Saved model features do not match the notebook pipeline. Train the model again."
            )
        trained = cls(
            name=payload["model_name"],
            estimator=payload["estimator"],
            medians={key: float(value) for key, value in payload["medians"].items()},
            threshold_f1=float(payload["threshold_f1"]),
            threshold_cost=float(payload["threshold_cost"]),
            applied_threshold=float(payload.get("applied_threshold", payload["threshold_f1"])),
            trained_at=payload["trained_at"],
            positive_weight=float(payload["positive_weight"]),
            validation_at_half=payload["validation_at_half"],
            test_at_f1=payload["test_at_f1"],
            overfit=payload["overfit"],
            importance=payload["importance"],
            best_params=_plain(payload.get("best_params") or {}),
            cv_score=float(payload["cv_score"]) if payload.get("cv_score") is not None else 0.0,
            insights=list(payload.get("insights") or []),
            train_proba=np.asarray([]),
            val_proba=np.asarray([]),
            test_proba=np.asarray([]),
        )
        trained.rescore(bundle)
        return trained

    def rescore(self, bundle: Bundle) -> None:
        self.train_proba = self.estimator.predict_proba(bundle.Xtr)[:, 1]
        self.val_proba = self.estimator.predict_proba(bundle.Xva)[:, 1]
        self.test_proba = self.estimator.predict_proba(bundle.Xte)[:, 1]
        self.validation_at_half = classification_report(bundle.yva, self.val_proba, 0.5)
        self.test_at_f1 = classification_report(bundle.yte, self.test_proba, self.threshold_f1)
        self.overfit = _overfit(bundle, self)

    def card(self, filename: str | None = None) -> dict[str, Any]:
        return {
            "filename": filename,
            "model_name": self.name,
            "trained_at": self.trained_at,
            "threshold_f1": self.threshold_f1,
            "threshold_cost": self.threshold_cost,
            "applied_threshold": self.applied_threshold,
            "validation_pr_auc": self.validation_at_half["pr_auc"],
            "validation_f1": self.validation_at_half["f1"],
            "test_f1": self.test_at_f1["f1"],
            "test_precision": self.test_at_f1["precision"],
            "test_recall": self.test_at_f1["recall"],
            "cv_score": self.cv_score,
            "best_params": self.best_params,
            "parameters": parameter_rows(self.best_params),
            "insights": self.insights,
            "note": SEARCH_NOTES[self.name],
        }


def _overfit(bundle: Bundle, trained: TrainedModel) -> dict[str, float]:
    train_f1 = float(f1_score(bundle.ytr, trained.train_proba >= 0.5))
    val_f1 = float(f1_score(bundle.yva, trained.val_proba >= 0.5))
    test_f1 = float(f1_score(bundle.yte, trained.test_proba >= 0.5))
    return {
        "train_f1": train_f1,
        "validation_f1": val_f1,
        "test_f1": test_f1,
        "train_validation_gap": train_f1 - val_f1,
    }


def _insights(name: str, params: dict[str, Any], cv_score: float, validation: dict, test: dict, overfit: dict, importance: list[dict]) -> list[str]:
    pretty = ", ".join(f"{row['name']} = {row['value']}" for row in parameter_rows(params)) or "the search defaults"
    top = ", ".join(item["label"] for item in importance[:3]) or "the engineered flags"
    return [
        f"Best settings from the training-split search: {pretty}. Mean cross-validated PR-AUC was {cv_score:.4f}. That score never saw the validation or test rows.",
        f"On the validation split, at threshold 0.5, PR-AUC is {validation['pr_auc']:.4f}, precision {validation['precision']:.1%} and recall {validation['recall']:.1%}.",
        f"On the untouched test split, at this model's max-F1 threshold {test['threshold']:.3f}, precision is {test['precision']:.1%}, recall is {test['recall']:.1%} and F1 is {test['f1']:.3f}.",
        f"Train F1 at 0.5 is {overfit['train_f1']:.3f} and validation F1 is {overfit['validation_f1']:.3f}. The gap is {overfit['train_validation_gap']:.3f}.",
        f"The largest {('coefficients' if name == 'Logistic Regression' else 'splits')} belong to {top}.",
        SEARCH_NOTES[name],
    ]


def train_one(bundle: Bundle, name: str, n_jobs: int = -1) -> TrainedModel:
    search = _search(name, bundle.positive_weight, n_jobs=n_jobs)
    search.fit(bundle.Xtr, bundle.ytr)
    estimator = search.best_estimator_
    best_params = _plain(search.best_params_)
    cv_score = float(search.best_score_)
    train_proba = estimator.predict_proba(bundle.Xtr)[:, 1]
    val_proba = estimator.predict_proba(bundle.Xva)[:, 1]
    test_proba = estimator.predict_proba(bundle.Xte)[:, 1]
    threshold_f1, threshold_cost = choose_thresholds(
        bundle.yva, val_proba, bundle.amount.loc[bundle.yva.index].to_numpy()
    )
    medians = {column: float(bundle.Xtr[column].median()) for column in FEATURE_COLUMNS}
    validation = classification_report(bundle.yva, val_proba, 0.5)
    test_report = classification_report(bundle.yte, test_proba, threshold_f1)
    importance = feature_importance(name, estimator, FEATURE_COLUMNS)
    trained = TrainedModel(
        name=name,
        estimator=estimator,
        medians=medians,
        threshold_f1=threshold_f1,
        threshold_cost=threshold_cost,
        applied_threshold=threshold_f1,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        positive_weight=bundle.positive_weight,
        validation_at_half=validation,
        test_at_f1=test_report,
        overfit={},
        importance=importance,
        best_params=best_params,
        cv_score=cv_score,
        insights=[],
        train_proba=train_proba,
        val_proba=val_proba,
        test_proba=test_proba,
    )
    trained.overfit = _overfit(bundle, trained)
    trained.insights = _insights(name, best_params, cv_score, validation, test_report, trained.overfit, importance)
    return trained


def preprocessing_report(bundle: Bundle) -> dict[str, Any]:
    flags = ["orig_emptied", "amount_eq_oldbal", "dest_zero", "orig_recon_ok", "dest_recon_ok", "risky_type", "emptied_risky"]
    fraud = bundle.y == 1
    flag_rates = []
    for name in flags:
        flag_rates.append(
            {
                "feature": name,
                "label": FEATURE_LABELS[name],
                "fraud": float(bundle.X.loc[fraud, name].mean()),
                "legit": float(bundle.X.loc[~fraud, name].mean()),
            }
        )
    groups = []
    for title, columns in FEATURE_GROUPS:
        groups.append(
            {
                "title": title,
                "features": [
                    {"name": column, "label": FEATURE_LABELS[column], "help": FEATURE_HELP[column]}
                    for column in columns
                ],
            }
        )
    summary = bundle.summary
    steps = [
        {
            "title": "Read the bundled file",
            "detail": f"{summary['rows']:,} transactions, {summary['fraud']:,} fraud. Missing values: {summary['missing']}. Duplicates: {summary['duplicates']}.",
        },
        {
            "title": "Drop the identifiers",
            "detail": "nameOrig and nameDest are account IDs with too many distinct values to treat as categories.",
        },
        {
            "title": "Leave step out",
            "detail": f"Legitimate rows stop at step {summary['legit_max_step']}. {summary['fraud_rows_after_legit_step']:,} fraud rows are later, so step leaks the label.",
        },
        {
            "title": "Log the skewed amounts",
            "detail": "Amount and the four balances are transformed with log(1 + value). The raw balances are kept as well.",
        },
        {
            "title": "Rebuild the ledger",
            "detail": "Sender and recipient balance changes, and the gap between that change and the amount, become errOrig and errDest.",
        },
        {
            "title": "Encode the type",
            "detail": "CASH_IN, CASH_OUT, DEBIT, PAYMENT and TRANSFER become five 0/1 columns. Risky type is TRANSFER or CASH_OUT.",
        },
        {
            "title": "Add the pattern flags",
            "detail": "Emptied account, amount equal to the old balance, recipient left at zero, ledger reconciles, and emptied plus risky type.",
        },
        {
            "title": "Split, and fit the scaler only on train",
            "detail": "Stratified 60% train, 20% validation, 20% test, seed 42. The logistic regression scaler is inside the training pipeline, so validation and test do not influence it.",
        },
    ]
    return {
        "rows_in": summary["rows"],
        "rows_out": int(len(bundle.X)),
        "raw_columns": 10,
        "model_features": len(FEATURE_COLUMNS),
        "dropped": ["step", "nameOrig", "nameDest"],
        "steps": steps,
        "groups": groups,
        "flag_rates": flag_rates,
        "split": summary["split"],
    }


def transactions_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "type": str(record["type"]),
                "amount": float(record["amount"]),
                "oldbalanceOrg": float(record["oldbalanceOrg"]),
                "newbalanceOrig": float(record["newbalanceOrig"]),
                "oldbalanceDest": float(record["oldbalanceDest"]),
                "newbalanceDest": float(record["newbalanceDest"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No transactions to score.")
    unknown = sorted(set(frame["type"]) - set(TYPES))
    if unknown:
        raise ValueError(f"Unsupported transaction type: {', '.join(unknown)}")
    return frame


def local_contributions(trained: TrainedModel, features: pd.DataFrame) -> list[dict[str, Any]]:
    base = float(trained.estimator.predict_proba(features)[:, 1][0])
    deltas = []
    for column in features.columns:
        altered = features.copy()
        altered[column] = trained.medians[column]
        altered_probability = float(trained.estimator.predict_proba(altered)[:, 1][0])
        deltas.append((column, base - altered_probability))
    deltas.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "feature": column,
            "label": FEATURE_LABELS.get(column, column),
            "delta": _json_float(delta, 4),
        }
        for column, delta in deltas[:8]
        if abs(delta) > 1e-6
    ]


def signal_lines(raw_row: pd.Series, features: pd.Series) -> list[str]:
    lines = []
    tx_type = str(raw_row["type"])
    if tx_type in {"TRANSFER", "CASH_OUT"}:
        lines.append(f"{tx_type} is one of the only two types that contain fraud in this file.")
    else:
        lines.append(f"{tx_type} has no fraud rows in this file.")
    if int(features["amount_eq_oldbal"]) == 1 and float(raw_row["oldbalanceOrg"]) > 0:
        lines.append("The amount equals the sender's entire old balance, the dominant fraud pattern here.")
    if int(features["orig_emptied"]) == 1:
        lines.append("The sender balance is left at exactly zero.")
    elif float(raw_row["oldbalanceOrg"]) > 0:
        lines.append("The sender account is not emptied.")
    if int(features["dest_zero"]) == 1:
        lines.append("The recipient balance stays at zero on both sides of the transaction.")
    if int(features["orig_recon_ok"]) == 1:
        lines.append("The sender ledger drops by the transaction amount.")
    else:
        lines.append("The sender ledger does not reconcile with the amount.")
    if int(features["dest_recon_ok"]) == 0:
        lines.append("The recipient ledger does not reconcile with the amount.")
    return lines


def risk_level(probability: float, threshold: float, is_fraud: bool) -> str:
    if is_fraud and probability >= 0.75:
        return "CRITICAL"
    if is_fraud:
        return "HIGH"
    if probability >= threshold * 0.5:
        return "MEDIUM"
    return "LOW"


def recommended_action(is_fraud: bool, probability: float, threshold: float) -> str:
    if is_fraud and probability >= max(0.75, threshold):
        return (
            "Queue this for urgent manual review before the funds settle. "
            "A model flag is not the same as an automatic block."
        )
    if is_fraud:
        return (
            "Send this to a review queue. Keep the auto-block threshold higher than the review threshold, "
            "because several legitimate transfers look similar."
        )
    if probability >= threshold * 0.5:
        return "Approve, and keep the account on a watch list. The score is below the decision threshold but not negligible."
    return "Approve. The score is below the decision threshold."


def score_frame(trained: TrainedModel, raw: pd.DataFrame, explain: bool = True) -> list[dict[str, Any]]:
    features = build_features(raw)
    probabilities = trained.estimator.predict_proba(features)[:, 1]
    threshold = trained.applied_threshold
    results = []
    for position, (index, row) in enumerate(raw.iterrows()):
        probability = float(probabilities[position])
        is_fraud = probability >= threshold
        feature_row = features.loc[index]
        signals = signal_lines(row, feature_row)
        contributions = local_contributions(trained, features.loc[[index]]) if explain else []
        level = risk_level(probability, threshold, is_fraud)
        verdict = "flagged as fraud" if is_fraud else "scored as legitimate"
        summary = (
            f"This {row['type']} of {float(row['amount']):,.0f} is {verdict} "
            f"at {probability:.2%} probability. The decision threshold is {threshold:.1%} "
            f"({trained.name})."
        )
        if signals:
            summary += " " + " ".join(signals)
        analyzed = {
            "errOrig": _json_float(feature_row["errOrig"], 2),
            "errDest": _json_float(feature_row["errDest"], 2),
            "log_amount": _json_float(feature_row["log_amount"], 4),
            "orig_emptied": float(feature_row["orig_emptied"]),
            "amount_eq_oldbal": float(feature_row["amount_eq_oldbal"]),
            "dest_zero": float(feature_row["dest_zero"]),
            "orig_recon_ok": float(feature_row["orig_recon_ok"]),
            "dest_recon_ok": float(feature_row["dest_recon_ok"]),
            "risky_type": float(feature_row["risky_type"]),
            "emptied_risky": float(feature_row["emptied_risky"]),
        }
        results.append(
            {
                "is_fraud": bool(is_fraud),
                "fraud_probability": _json_float(probability, 6),
                "decision_threshold": float(threshold),
                "risk_level": level,
                "model_name": trained.name,
                "summary_text": summary,
                "recommended_action": recommended_action(is_fraud, probability, threshold),
                "signals": signals,
                "features_analyzed": analyzed,
                "feature_contributions": {item["label"]: item["delta"] for item in contributions},
                "contributions": contributions,
            }
        )
    return results


def evaluation_payload(bundle: Bundle, trained: TrainedModel) -> dict[str, Any]:
    amounts = bundle.amount.loc[bundle.yte.index].to_numpy()
    headline_metrics = classification_report(
        bundle.yte, trained.test_proba, trained.applied_threshold
    )
    money = financials(bundle.yte, trained.test_proba, amounts, trained.applied_threshold)
    headline = {**headline_metrics, **money}
    fpr, tpr, _ = roc_curve(bundle.yte, trained.test_proba)
    precision, recall, _ = precision_recall_curve(bundle.yte, trained.test_proba)
    rows = []
    for index, probability in zip(bundle.Xte.index, trained.test_proba):
        raw = bundle.raw.loc[index]
        rows.append(
            {
                "id": int(index),
                "type": str(raw["type"]),
                "amount": float(raw["amount"]),
                "oldbalanceOrg": float(raw["oldbalanceOrg"]),
                "newbalanceOrig": float(raw["newbalanceOrig"]),
                "oldbalanceDest": float(raw["oldbalanceDest"]),
                "newbalanceDest": float(raw["newbalanceDest"]),
                "actual": int(bundle.y.loc[index]),
                "probability": float(probability),
            }
        )
    n_fraud = int(bundle.yte.sum())
    findings = [
        (
            f"On {len(bundle.yte):,} untouched test transactions, {trained.name} at threshold "
            f"{trained.applied_threshold:.3f} has precision {headline['precision']:.2%}, "
            f"recall {headline['recall']:.2%}, F1 {headline['f1']:.3f}, "
            f"ROC-AUC {headline['roc_auc']:.4f} and PR-AUC {headline['pr_auc']:.4f}."
        ),
        (
            f"Confusion matrix: {headline['tn']:,} true negatives, {headline['fp']:,} false positives, "
            f"{headline['fn']:,} false negatives, {headline['tp']:,} true positives."
        ),
        (
            f"Missed fraud value is {headline['missed_loss']:,.0f} of {headline['fraud_value']:,.0f}. "
            f"Reviewing every alert at {INVESTIGATION_COST:.0f} each costs {headline['review_cost']:,.0f}. "
            f"Illustrative profit versus approving everything is {headline['uplift']:,.0f} on this test split. "
            "Fee revenue is counted once, only on legitimate transactions that are approved."
        ),
        (
            f"At a fixed threshold of 0.5, train F1 is {trained.overfit['train_f1']:.3f} and validation F1 is "
            f"{trained.overfit['validation_f1']:.3f} (gap {trained.overfit['train_validation_gap']:.3f}). "
            f"Test F1 at 0.5 is {trained.overfit['test_f1']:.3f}."
        ),
        (
            "The scorer is the notebook's post-transaction model: it uses the balances after the transaction. "
            "step, nameOrig and nameDest are not features."
        ),
        (
            f"The test split has {n_fraud} frauds, so one missed fraud moves recall by {1 / n_fraud:.2%}. "
            "Treat the percentages as results on this sample, not as a production guarantee."
        ),
    ]
    return {
        "model_name": trained.name,
        "trained_at": trained.trained_at,
        "scenario": "post-transaction",
        "applied_threshold": trained.applied_threshold,
        "threshold_f1": trained.threshold_f1,
        "threshold_cost": trained.threshold_cost,
        "assumptions": {
            "investigation_cost": INVESTIGATION_COST,
            "fee": FEE,
            "currency": "unspecified units, as in the source file",
        },
        "split": bundle.summary["split"],
        "validation_at_half": trained.validation_at_half,
        "headline": headline,
        "overfit": trained.overfit,
        "importance": trained.importance,
        "best_params": trained.best_params,
        "parameters": parameter_rows(trained.best_params),
        "cv_score": trained.cv_score,
        "insights": trained.insights,
        "roc": _curve_points(fpr, tpr),
        "pr": _curve_points(recall, precision),
        "rows": rows,
        "findings": findings,
        "positive_weight": trained.positive_weight,
    }


def metrics_document(bundle: Bundle, trained: TrainedModel) -> dict[str, Any]:
    payload = evaluation_payload(bundle, trained)
    headline = payload["headline"]
    return {
        "model_loaded": True,
        "model_name": trained.name,
        "optimal_threshold": trained.threshold_f1,
        "applied_threshold": trained.applied_threshold,
        "pr_auc": headline["pr_auc"],
        "roc_auc": headline["roc_auc"],
        "precision": headline["precision"],
        "recall": headline["recall"],
        "f1": headline["f1"],
        "accuracy": headline["accuracy"],
        "confusion": {
            "tn": headline["tn"],
            "fp": headline["fp"],
            "fn": headline["fn"],
            "tp": headline["tp"],
        },
    }
