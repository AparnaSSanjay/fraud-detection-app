"""FastAPI service for the notebook fraud model.

Train, save, load and score against the bundled PaySim sample. The page is
served from this process so the charts and the API share one origin.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import re
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import pipeline

logger = logging.getLogger("fraud_intel")
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REGISTRY_PATH = MODEL_DIR / "registry.json"
FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.joblib$")
LOCK = threading.Lock()

STATE: dict[str, Any] = {
    "bundle": None,
    "dataset_error": None,
    "active": None,
    "active_file": None,
    "model_error": None,
    "comparison": [],
    "train_job": {"id": None, "status": "idle", "message": "", "completed": 0, "total": 0, "result": None},
}


def _read_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"active": None, "comparison": [], "models": {}}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except json.JSONDecodeError:
        logger.exception("Could not read model registry")
        return {"active": None, "comparison": [], "models": {}}


def _write_registry(registry: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if name != filename or not FILENAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Model filename must use letters, numbers, underscores or hyphens and end in .joblib.",
        )
    return name


def _model_path(filename: str) -> Path:
    path = (MODEL_DIR / _safe_filename(filename)).resolve()
    if path.parent != MODEL_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid model path.")
    return path


def _meta_from_trained(trained: pipeline.TrainedModel, filename: str) -> dict[str, Any]:
    card = trained.card(filename)
    return card


def _remember(filename: str, trained: pipeline.TrainedModel, make_active: bool) -> None:
    registry = _read_registry()
    registry.setdefault("models", {})
    registry["models"][filename] = _meta_from_trained(trained, filename)
    if make_active:
        registry["active"] = filename
        STATE["active"] = trained
        STATE["active_file"] = filename
        STATE["model_error"] = None
    _write_registry(registry)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "estimator" not in payload:
        raise pipeline.ModelNotReady(f"{path.name} is not a fraud model artifact from this app.")
    return payload


def activate_file(filename: str) -> pipeline.TrainedModel:
    path = _model_path(filename)
    name = path.name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Saved model {name} was not found.")
    bundle = _bundle()
    try:
        trained = pipeline.TrainedModel.from_artifact(_load_payload(path), bundle)
    except Exception as exc:
        logger.exception("Failed to load %s", name)
        raise HTTPException(status_code=400, detail=f"Could not load {name}: {exc}") from exc
    _remember(name, trained, make_active=True)
    return trained


def _autoload() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    registry = _read_registry()
    known = set(registry.get("models", {}))
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        if path.name in known:
            continue
        try:
            payload = _load_payload(path)
            trained = pipeline.TrainedModel.from_artifact(payload, STATE["bundle"])
            registry.setdefault("models", {})[path.name] = _meta_from_trained(trained, path.name)
        except Exception:
            logger.exception("Skipping unreadable model file %s", path.name)
    _write_registry(registry)
    STATE["comparison"] = registry.get("comparison") or []
    active_name = registry.get("active")
    if not active_name or not (MODEL_DIR / active_name).exists():
        ranked = [
            item for item in registry.get("models", {}).values()
            if item.get("filename") and (MODEL_DIR / item["filename"]).exists()
        ]
        if ranked:
            active_name = max(ranked, key=lambda item: item.get("validation_pr_auc") or 0)["filename"]
            registry["active"] = active_name
            _write_registry(registry)
    if active_name and (MODEL_DIR / active_name).exists():
        try:
            activate_file(active_name)
            return
        except HTTPException as exc:
            STATE["model_error"] = str(exc.detail)
    STATE["active"] = None
    STATE["active_file"] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    try:
        STATE["bundle"] = pipeline.make_bundle()
        STATE["dataset_error"] = None
        logger.info("Loaded dataset %s (%s rows)", STATE["bundle"].path, len(STATE["bundle"].raw))
        _autoload()
    except Exception as exc:
        logger.exception("Startup failed")
        STATE["dataset_error"] = str(exc)
    yield


app = FastAPI(
    title="Fraud Intelligence Engine",
    description="Train, save and score the PaySim fraud notebook on its bundled dataset.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionInput(BaseModel):
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)
    step: Optional[int] = Field(default=None, ge=1)


class BatchTransactionInput(BaseModel):
    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=20000)


class TrainRequest(BaseModel):
    models: list[Literal["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]] = Field(
        ..., min_length=1
    )


class FilenameRequest(BaseModel):
    filename: str


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., gt=0, lt=1)


def _bundle() -> pipeline.Bundle:
    if STATE["dataset_error"]:
        raise HTTPException(status_code=503, detail=STATE["dataset_error"])
    bundle = STATE["bundle"]
    if bundle is None:
        raise HTTPException(status_code=503, detail="Dataset is not loaded.")
    return bundle


def _active() -> pipeline.TrainedModel:
    _bundle()
    trained = STATE["active"]
    if trained is None:
        detail = STATE["model_error"] or "No model is loaded. Train one from the Train view, or load a saved file."
        raise HTTPException(status_code=409, detail=detail)
    return trained


def _health_body() -> dict[str, Any]:
    bundle = STATE["bundle"]
    trained = STATE["active"]
    return {
        "status": "healthy" if bundle is not None and STATE["dataset_error"] is None else "unavailable",
        "model_loaded": trained is not None,
        "model_name": trained.name if trained else None,
        "model_file": STATE["active_file"],
        "applied_threshold": trained.applied_threshold if trained else None,
        "threshold_f1": trained.threshold_f1 if trained else None,
        "threshold_cost": trained.threshold_cost if trained else None,
        "dataset_rows": int(len(bundle.raw)) if bundle is not None else 0,
        "dataset_path": str(bundle.path) if bundle is not None else None,
        "dataset_error": STATE["dataset_error"],
        "model_error": STATE["model_error"],
        "feature_count": len(pipeline.FEATURE_COLUMNS),
        "scenario": "post-transaction",
    }


@app.get("/")
def index_page():
    page = ROOT / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="index.html is missing.")
    return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/v1/health")
def health_check():
    return _health_body()


@app.get("/api/v1/metadata")
def metadata():
    trained = STATE["active"]
    return {
        "model_name": trained.name if trained else None,
        "model_loaded": trained is not None,
        "applied_threshold": trained.applied_threshold if trained else None,
        "threshold_f1": trained.threshold_f1 if trained else None,
        "threshold_cost": trained.threshold_cost if trained else None,
        "supported_transaction_types": pipeline.TYPES,
        "required_features": [
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ],
        "model_features": pipeline.FEATURE_COLUMNS,
        "excluded_features": ["step", "nameOrig", "nameDest"],
        "scenario": "post-transaction",
        "assumptions": {
            "investigation_cost": pipeline.INVESTIGATION_COST,
            "fee": pipeline.FEE,
        },
        "hyperparameters": pipeline.SEARCH_NOTES,
        "search_draws": pipeline.SEARCH_DRAWS,
        "search_folds": 3,
        "hand_cases": pipeline.HAND_CASES,
        "split": "stratified 60/20/20, random_state 42, matching the notebook",
    }


@app.get("/api/v1/metrics")
def metrics():
    return pipeline.metrics_document(_bundle(), _active())


@app.get("/api/v1/dataset/summary")
def dataset_summary():
    return _bundle().summary


@app.get("/api/v1/preprocessing")
def preprocessing():
    return pipeline.preprocessing_report(_bundle())


@app.get("/api/v1/models")
def list_models():
    registry = _read_registry()
    models = []
    for filename, meta in sorted(registry.get("models", {}).items()):
        if not (MODEL_DIR / filename).exists():
            continue
        item = dict(meta)
        item["filename"] = filename
        item["active"] = filename == STATE["active_file"]
        models.append(item)
    return {
        "active": STATE["active_file"],
        "models": models,
        "comparison": STATE["comparison"] or registry.get("comparison") or [],
    }


def _comparison_row(trained: pipeline.TrainedModel, card: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": trained.name,
        "filename": card["filename"],
        "accuracy": trained.validation_at_half["accuracy"],
        "precision": trained.validation_at_half["precision"],
        "recall": trained.validation_at_half["recall"],
        "f1": trained.validation_at_half["f1"],
        "roc_auc": trained.validation_at_half["roc_auc"],
        "pr_auc": trained.validation_at_half["pr_auc"],
        "threshold_f1": trained.threshold_f1,
        "threshold_cost": trained.threshold_cost,
        "test_f1": trained.test_at_f1["f1"],
        "test_precision": trained.test_at_f1["precision"],
        "test_recall": trained.test_at_f1["recall"],
        "cv_score": trained.cv_score,
        "best_params": trained.best_params,
        "parameters": card["parameters"],
        "insights": trained.insights,
    }


def _publish_job(job_id: str, **fields: Any) -> None:
    with LOCK:
        current = STATE["train_job"]
        if current.get("id") != job_id:
            return
        STATE["train_job"] = {**current, **fields}


def _fit_model(bundle: pipeline.Bundle, name: str) -> pipeline.TrainedModel:
    # One core inside the process. A second job pool here shuts the others down.
    return pipeline.train_one(bundle, name, n_jobs=1)


def _store_trained(trained: pipeline.TrainedModel) -> dict[str, Any]:
    filename = trained.name.lower().replace(" ", "_") + ".joblib"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained.to_artifact(), MODEL_DIR / filename)
    card = trained.card(filename)
    with LOCK:
        registry = _read_registry()
        registry.setdefault("models", {})[filename] = card
        _write_registry(registry)
    return card


def _run_train(job_id: str, requested: list[str]) -> None:
    bundle = STATE["bundle"]
    total = len(requested)
    done: dict[str, tuple[pipeline.TrainedModel, dict[str, Any]]] = {}
    failed: dict[str, str] = {}
    pending = list(requested)

    def progress(message: str, completed: int) -> None:
        _publish_job(
            job_id,
            status="running",
            message=message,
            completed=completed,
            total=total,
            running=[name for name in requested if name not in done and name not in failed],
            finished=list(done),
            result=None,
        )

    try:
        names = ", ".join(requested)
        progress(
            f"Training {total} selected model(s) at the same time: {names}.",
            0,
        )
        if total == 1:
            executor = ThreadPoolExecutor(max_workers=1)
            launch = lambda pool, name: pool.submit(pipeline.train_one, bundle, name, -1)
        else:
            executor = ProcessPoolExecutor(
                max_workers=total,
                mp_context=multiprocessing.get_context("spawn"),
            )
            launch = lambda pool, name: pool.submit(_fit_model, bundle, name)
        with executor as pool:
            futures = {launch(pool, name): name for name in requested}
            for future in as_completed(futures):
                name = futures[future]
                pending.remove(name)
                try:
                    trained = future.result()
                except Exception as exc:
                    logger.exception("Tuning failed for %s", name)
                    failed[name] = str(exc)
                    still = ", ".join(pending) or "none"
                    progress(f"{name} failed. Still training: {still}.", len(done))
                    continue
                card = _store_trained(trained)
                done[name] = (trained, card)
                logger.info("Finished %s", name)
                still = ", ".join(pending) or "none"
                progress(
                    f"{name} finished ({len(done)}/{total}). Still training: {still}.",
                    len(done),
                )
        if not done:
            detail = "; ".join(f"{name}: {error}" for name, error in failed.items())
            _publish_job(
                job_id,
                status="failed",
                message=f"Tuning failed. {detail}",
                completed=0,
                total=total,
                running=[],
                finished=[],
                result=None,
            )
            return
        ordered = [(done[name][0], done[name][1]) for name in requested if name in done]
        best, best_card = max(ordered, key=lambda item: item[0].validation_at_half["pr_auc"])
        comparison = [_comparison_row(trained, card) for trained, card in ordered]
        with LOCK:
            registry = _read_registry()
            registry.setdefault("models", {})
            for _, card in ordered:
                registry["models"][card["filename"]] = card
            registry["active"] = best_card["filename"]
            registry["comparison"] = comparison
            _write_registry(registry)
            STATE["comparison"] = comparison
            STATE["active"] = best
            STATE["active_file"] = best_card["filename"]
            STATE["model_error"] = None
        if failed:
            detail = "; ".join(f"{name}: {error}" for name, error in failed.items())
            message = (
                f"Saved {len(ordered)} model(s). {best.name} is selected for scoring. "
                f"Failed: {detail}"
            )
            status = "failed"
        else:
            message = (
                f"Tuned {len(ordered)} model(s) in parallel. {best.name} is selected for scoring "
                f"because its validation PR-AUC is the highest."
            )
            status = "complete"
        _publish_job(
            job_id,
            status=status,
            message=message,
            completed=len(done),
            total=total,
            current_model=best.name,
            running=[],
            finished=[trained.name for trained, _card in ordered],
            result={
                "active_model": best.name,
                "active_file": best_card["filename"],
                "saved": [card for _, card in ordered],
                "comparison": comparison,
                "failed": failed,
                "message": message,
            },
        )
    except Exception as exc:
        logger.exception("Tuning failed")
        _publish_job(
            job_id,
            status="failed",
            message=f"Tuning failed: {exc}",
            completed=len(done),
            total=total,
            running=[],
            result=None,
        )


@app.post("/api/v1/train")
def train_models(body: TrainRequest):
    _bundle()
    if STATE["train_job"].get("status") == "running":
        raise HTTPException(status_code=409, detail="A tuning run is already in progress.")
    requested = list(dict.fromkeys(body.models))
    job_id = uuid.uuid4().hex[:12]
    names = ", ".join(requested)
    STATE["train_job"] = {
        "id": job_id,
        "status": "running",
        "message": f"Training {len(requested)} selected model(s) at the same time: {names}.",
        "completed": 0,
        "total": len(requested),
        "current_model": None,
        "running": requested,
        "finished": [],
        "result": None,
    }
    threading.Thread(target=_run_train, args=(job_id, requested), daemon=True).start()
    return {"job_id": job_id, "status": "running", "total": len(requested)}


@app.get("/api/v1/train/status")
def train_status():
    return STATE["train_job"]


@app.post("/api/v1/models/save")
def save_copy(body: FilenameRequest):
    trained = _active()
    raw_name = body.filename if body.filename.endswith(".joblib") else f"{body.filename}.joblib"
    path = _model_path(raw_name)
    with LOCK:
        joblib.dump(trained.to_artifact(), path)
        _remember(path.name, trained, make_active=False)
    return {"filename": path.name, "model_name": trained.name}


@app.post("/api/v1/models/load")
def load_model(body: FilenameRequest):
    with LOCK:
        trained = activate_file(body.filename)
    return {"filename": STATE["active_file"], "model_name": trained.name, "card": trained.card(STATE["active_file"])}


def _activate_best_saved() -> None:
    registry = _read_registry()
    ranked = [
        item for item in registry.get("models", {}).values()
        if item.get("filename") and (MODEL_DIR / item["filename"]).exists()
    ]
    if not ranked:
        return
    best = max(ranked, key=lambda item: item.get("validation_pr_auc") or 0)
    activate_file(best["filename"])


@app.delete("/api/v1/models/{filename}")
def delete_model(filename: str):
    path = _model_path(filename)
    name = path.name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Saved model {name} was not found.")
    with LOCK:
        path.unlink()
        registry = _read_registry()
        registry.get("models", {}).pop(name, None)
        removed_active = registry.get("active") == name or STATE["active_file"] == name
        if removed_active:
            registry["active"] = None
            STATE["active"] = None
            STATE["active_file"] = None
        _write_registry(registry)
        if removed_active:
            _activate_best_saved()
    return {"deleted": name, "active": STATE["active_file"]}


@app.post("/api/v1/threshold")
def set_threshold(body: ThresholdRequest):
    trained = _active()
    with LOCK:
        trained.applied_threshold = float(body.threshold)
        if STATE["active_file"]:
            path = MODEL_DIR / STATE["active_file"]
            if path.exists():
                joblib.dump(trained.to_artifact(), path)
                _remember(STATE["active_file"], trained, make_active=True)
    return {
        "applied_threshold": trained.applied_threshold,
        "model_name": trained.name,
        "metrics": pipeline.metrics_document(_bundle(), trained),
    }


@app.get("/api/v1/evaluation")
def evaluation():
    return pipeline.evaluation_payload(_bundle(), _active())


@app.post("/api/v1/predict")
def predict_fraud(data: TransactionInput):
    trained = _active()
    frame = pipeline.transactions_frame([data.model_dump()])
    return pipeline.score_frame(trained, frame, explain=True)[0]


@app.post("/api/v1/predict/batch")
def predict_batch(body: BatchTransactionInput):
    trained = _active()
    records = [item.model_dump() for item in body.transactions]
    frame = pipeline.transactions_frame(records)
    explain = len(records) <= 30
    results = pipeline.score_frame(trained, frame, explain=explain)
    return {
        "model_name": trained.name,
        "decision_threshold": trained.applied_threshold,
        "count": len(results),
        "flagged": sum(1 for item in results if item["is_fraud"]),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
