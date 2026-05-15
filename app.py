from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ml_system.service import MLSystem

ROOT = Path(__file__).resolve().parent
service = MLSystem(ROOT)

app = FastAPI(title="Online Shoppers ML System", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@app.on_event("startup")
def startup() -> None:
    service.load()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": service.current_version}


@app.get("/api/versions")
def versions() -> dict[str, Any]:
    return {"available_versions": service.available_versions(), "current_version": service.current_version}


@app.post("/api/version/{version}")
def switch_version(version: str) -> dict[str, Any]:
    if version not in service.available_versions():
        return {"ok": False, "message": f"Version {version} not found."}
    service.load(version=version)
    return {"ok": True, "current_version": service.current_version}


@app.post("/api/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    threshold = float(payload.pop("threshold", 0.5))
    result = service.predict_one(payload, threshold=threshold)
    return {
        "purchase_probability": round(result.probability, 6),
        "decision": result.decision,
        "threshold": round(result.threshold, 2),
        "latency_ms": round(result.latency_ms, 3),
    }


@app.post("/api/predict-batch")
def predict_batch(request_rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    rows = []
    for row in request_rows:
        result = service.predict_one(dict(row), threshold=threshold)
        scored = dict(row)
        scored["purchase_probability"] = round(result.probability, 6)
        scored["decision"] = result.decision
        rows.append(scored)
    return {"count": len(rows), "rows": rows}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    metrics = service.summary_metrics()
    drift_top = (
        service.drift_report.sort_values("psi", ascending=False).head(10).to_dict(orient="records")
        if not service.drift_report.empty
        else []
    )
    target_counts = {
        "purchase": int(round(metrics["rows"] * metrics["positive_rate"] / 100.0)),
        "no_purchase": int(metrics["rows"] - round(metrics["rows"] * metrics["positive_rate"] / 100.0)),
    }
    raw_df = service._load_raw()
    return templates.TemplateResponse(
        name="index.html",
        context={
            "request": request,
            "metrics": metrics,
            "drift_top_json": json.dumps(drift_top),
            "target_counts_json": json.dumps(target_counts),
            "versions": service.available_versions(),
            "month_options": service.MONTH_OPTIONS,
            "visitor_types": sorted(raw_df["VisitorType"].dropna().unique().tolist()),
            "os_options": sorted(raw_df["OperatingSystems"].dropna().unique().tolist()),
            "browser_options": sorted(raw_df["Browser"].dropna().unique().tolist()),
            "region_options": sorted(raw_df["Region"].dropna().unique().tolist()),
            "traffic_options": sorted(raw_df["TrafficType"].dropna().unique().tolist()),
        },
    )


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    Administrative: int = Form(...),
    Administrative_Duration: float = Form(...),
    Informational: int = Form(...),
    Informational_Duration: float = Form(...),
    ProductRelated: int = Form(...),
    ProductRelated_Duration: float = Form(...),
    BounceRates: float = Form(...),
    ExitRates: float = Form(...),
    PageValues: float = Form(...),
    SpecialDay: float = Form(...),
    OperatingSystems: int = Form(...),
    Browser: int = Form(...),
    Region: int = Form(...),
    TrafficType: int = Form(...),
    VisitorType: str = Form(...),
    Weekend: int = Form(...),
    Month: str = Form(...),
    threshold: float = Form(0.5),
) -> HTMLResponse:
    payload = {
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "OperatingSystems": OperatingSystems,
        "Browser": Browser,
        "Region": Region,
        "TrafficType": TrafficType,
        "VisitorType": VisitorType,
        "Weekend": Weekend,
        "Month": Month,
    }
    result = service.predict_one(payload, threshold=threshold)

    metrics = service.summary_metrics()
    drift_top = (
        service.drift_report.sort_values("psi", ascending=False).head(10).to_dict(orient="records")
        if not service.drift_report.empty
        else []
    )
    target_counts = {
        "purchase": int(round(metrics["rows"] * metrics["positive_rate"] / 100.0)),
        "no_purchase": int(metrics["rows"] - round(metrics["rows"] * metrics["positive_rate"] / 100.0)),
    }
    raw_df = service._load_raw()

    return templates.TemplateResponse(
        name="index.html",
        context={
            "request": request,
            "metrics": metrics,
            "drift_top_json": json.dumps(drift_top),
            "target_counts_json": json.dumps(target_counts),
            "versions": service.available_versions(),
            "prediction": {
                "probability": f"{result.probability:.2%}",
                "decision": result.decision,
                "latency_ms": f"{result.latency_ms:.2f}",
                "threshold": f"{threshold:.2f}",
            },
            "month_options": service.MONTH_OPTIONS,
            "visitor_types": sorted(raw_df["VisitorType"].dropna().unique().tolist()),
            "os_options": sorted(raw_df["OperatingSystems"].dropna().unique().tolist()),
            "browser_options": sorted(raw_df["Browser"].dropna().unique().tolist()),
            "region_options": sorted(raw_df["Region"].dropna().unique().tolist()),
            "traffic_options": sorted(raw_df["TrafficType"].dropna().unique().tolist()),
        },
    )
