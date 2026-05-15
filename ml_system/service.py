from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PredictionResult:
    probability: float
    decision: str
    threshold: float
    latency_ms: float


class MLSystem:
    REQUIRED_INPUT_COLUMNS = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
        "Month",
    ]

    MODEL_INPUT_COLUMNS = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
        "TotalSessionDuration",
        "ProductInfoRatio",
        "EngagementScore",
        "Month_sin",
        "Month_cos",
    ]

    CATEGORICAL_COLUMNS = ["VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
    BOOL_MAP = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}
    MONTH_OPTIONS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTH_OPTIONS)}

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifact_root = root / "artifacts"
        self.raw_path = root / "online_shoppers_intention.csv"
        self._raw_df: pd.DataFrame | None = None
        self._target: pd.Series | None = None
        self.metadata: dict[str, Any] = {}
        self.drift_report: pd.DataFrame = pd.DataFrame()
        self.processed_columns: list[str] = []
        self.current_version: str = ""
        self.preprocessor: ColumnTransformer | None = None
        self.model: LogisticRegression | None = None
        self.model_metrics: dict[str, float] = {}

    @staticmethod
    def _version_key(name: str) -> tuple[int, str]:
        if name.lower().startswith("v") and name[1:].isdigit():
            return (int(name[1:]), name)
        return (-1, name)

    def available_versions(self) -> list[str]:
        versions = [d.name for d in self.artifact_root.iterdir() if d.is_dir()]
        versions.sort(key=self._version_key)
        return versions

    def _load_version_files(self, version: str) -> None:
        artifact_dir = self.artifact_root / version
        metadata_path = artifact_dir / "metadata.json"
        drift_path = artifact_dir / "drift_report.csv"
        y_path = artifact_dir / "y.csv"
        x_path = artifact_dir / "X_processed.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file for version {version}: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.drift_report = pd.read_csv(drift_path) if drift_path.exists() else pd.DataFrame()
        self._target = pd.read_csv(y_path).iloc[:, 0] if y_path.exists() else None
        if x_path.exists():
            self.processed_columns = pd.read_csv(x_path, nrows=1).columns.tolist()

    def _load_raw(self) -> pd.DataFrame:
        if self._raw_df is None:
            self._raw_df = pd.read_csv(self.raw_path)
        return self._raw_df.copy()

    def _engineer_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        if "Weekend" in df.columns:
            df["Weekend"] = df["Weekend"].map(self.BOOL_MAP).fillna(0).astype(int)

        df["Month_num"] = df["Month"].map(self.MONTH_TO_NUM)
        if df["Month_num"].isna().any():
            df["Month_num"] = df["Month_num"].fillna(df["Month_num"].mode().iloc[0])

        df["TotalSessionDuration"] = (
            df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
        )
        df["ProductInfoRatio"] = df["ProductRelated"] / (df["Informational"] + 1)
        df["EngagementScore"] = df["PageValues"] / (df["TotalSessionDuration"] + 1)
        df["Month_sin"] = np.sin(2 * np.pi * df["Month_num"] / 12.0)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month_num"] / 12.0)
        df = df.drop(columns=["Month", "Month_num"], errors="ignore")
        return df[self.MODEL_INPUT_COLUMNS].copy()

    def _fit_preprocessor(self, x_df: pd.DataFrame) -> ColumnTransformer:
        num_cols = [c for c in x_df.columns if c not in self.CATEGORICAL_COLUMNS]
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, self.CATEGORICAL_COLUMNS),
            ],
            remainder="drop",
        )
        preprocessor.fit(x_df)
        return preprocessor

    def load(self, version: str | None = None) -> None:
        versions = self.available_versions()
        if not versions:
            raise FileNotFoundError("No versions found in artifacts/ directory.")
        self.current_version = version or versions[-1]
        self._load_version_files(self.current_version)

        raw = self._load_raw()
        x_df = self._engineer_features(raw)
        self.preprocessor = self._fit_preprocessor(x_df)

        transformed = self.preprocessor.transform(x_df)
        transformed_df = pd.DataFrame(transformed, columns=self._feature_names(x_df), index=x_df.index)

        y = self._target
        if y is None or len(y) != len(transformed_df):
            y = raw["Revenue"].map(self.BOOL_MAP).fillna(0).astype(int)
            self._target = y

        x_train, x_test, y_train, y_test = train_test_split(
            transformed_df,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.model = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.model.fit(x_train, y_train)

        pred = self.model.predict(x_test)
        prob = self.model.predict_proba(x_test)[:, 1]
        self.model_metrics = {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
        }

    def _feature_names(self, x_df: pd.DataFrame) -> list[str]:
        assert self.preprocessor is not None
        num_cols = [c for c in x_df.columns if c not in self.CATEGORICAL_COLUMNS]
        encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = encoder.get_feature_names_out(self.CATEGORICAL_COLUMNS).tolist()
        return num_cols + cat_names

    def predict_one(self, payload: dict[str, Any], threshold: float = 0.5) -> PredictionResult:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model system is not loaded.")

        input_df = pd.DataFrame([payload])
        missing = [c for c in self.REQUIRED_INPUT_COLUMNS if c not in input_df.columns]
        if missing:
            raise ValueError(f"Missing input fields: {missing}")

        frame = self._engineer_features(input_df[self.REQUIRED_INPUT_COLUMNS])

        t0 = time.perf_counter()
        transformed = self.preprocessor.transform(frame)
        probability = float(self.model.predict_proba(transformed)[0, 1])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        decision = "purchase" if probability >= threshold else "no_purchase"
        return PredictionResult(
            probability=probability,
            decision=decision,
            threshold=threshold,
            latency_ms=latency_ms,
        )

    def summary_metrics(self) -> dict[str, Any]:
        rows = int(self.metadata.get("total_rows", len(self._raw_df) if self._raw_df is not None else 0))
        out_features = int(self.metadata.get("total_features_after_encoding", len(self.processed_columns)))
        positive_rate = float(self.metadata.get("positive_rate", 0.0))
        high_drift = int(self.metadata.get("features_with_high_drift", 0))
        return {
            "rows": rows,
            "features": out_features,
            "positive_rate": positive_rate,
            "high_drift": high_drift,
            "version": self.current_version,
            **self.model_metrics,
        }
