import json
import time
import urllib.request
from typing import Optional
 
import pandas as pd
import requests
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


DEMO_API_URL = "https://simple-jev-demo-api.featherless.ai/v1/classifier"
DEMO_MODELS_URL = "https://simple-jev-demo-api.featherless.ai/v1/models"
# PROD_API_URL = "https://api.featherless.ai/v1/classifier"
# DEFAULT_DEMO_MODEL = "featherless-ai/gemma-4-26B-A4B-classifier"
# DEFAULT_DEMO_MODEL = "featherless-ai/Qwen3.8-27B-classifier"
DEFAULT_DEMO_MODEL = "featherless-ai/Qwen3.6-35B-A3B-classifier"

 
DEMO_RATE_LIMIT_RPS = 4
TARGET_SEQ_CHAR_BUDGET = 300

def truncate_seq(seq: str, max_chars: int) -> str:
    if len(seq) <= max_chars:
        return seq
    head = max_chars // 2
    tail = max_chars - head - 5
    return f"{seq[:head]}...{seq[-tail:]}"


def build_request(
    row: pd.Series,
    model: str,
    target_seq_char_budget: int = TARGET_SEQ_CHAR_BUDGET,
) -> dict:
    """
    One /v1/classifier request body for a single candidate design.
 
    `question_type="noul"` instead asks a yes/no proposition ("does it
    bind?") and returns a single 0.01-0.99 score -- useful if you want a
    continuous score for an ROC curve without reading into a 2-class
    `choice` response.
    """
    # state = {
    #     "target_name": row["target"],
    #     "generator": row["generator"],
    #     # "sequence_design_method": row["sequence_design_method"],
    #     # "sequence": truncate_seq(str(row["sequence"]), target_seq_char_budget),
    #     "sequence": row["sequence"],
    #     "ipsae_min_boltz2": row["ipsae_min_boltz2"],
    #     "sc_dockq_boltz2": row["sc_dockq_boltz2"],
    #     "epitope_n_residues": row["epitope_n_residues"],
    # }

    ## Generalize state generation for all columns in the row except uuid and y_true
    state = {k: v for k, v in row.items() if k not in ["uuid", "y_true"]}
 
    question = {
        "type": "noul",
        "instructions": "Does the candidate protein bind the target protein?",
        "criteria": {
            "true": "The candidate binds the target protein.",
            "false": "The candidate does not bind the target protein.",
        },
    }
 
    return {"model": model, "state": state, "questions": {"binding": question}}

def list_available_models(models_url: str = DEMO_MODELS_URL) -> list[str]:
    with urllib.request.urlopen(models_url, timeout=30) as resp:
        data = json.load(resp)
    return [m["id"] for m in data.get("data", [])]
 
 
def call_classifier(
    payload: dict,
    api_url: str = DEMO_API_URL,
    api_key: Optional[str] = None,
    timeout: float = 45.0,
    max_retries: int = 3,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
 
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            response = exc.response
            if response.status_code == 429 and attempt < max_retries:
                retry_after = float(response.headers.get("Retry-After", 1.0))
                time.sleep(retry_after)
                continue
            raise RuntimeError(
                f"Simple Jev API error {response.status_code}: {response.text}"
            ) from exc
    raise RuntimeError("Simple Jev API: exhausted retries on HTTP 429")


def classify_dataframe(
    df: pd.DataFrame,
    model: str = DEFAULT_DEMO_MODEL,
    api_url: str = DEMO_API_URL,
    api_key: Optional[str] = None,
    rps: float = DEMO_RATE_LIMIT_RPS,
) -> pd.DataFrame:
    delay = 1.0 / rps
    rows = []
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        payload = build_request(row, model=model)
        record = {"uuid": row["uuid"]}
        try:
            resp = call_classifier(payload, api_url=api_url, api_key=api_key)
            ans = resp["answers"]["binding"]
            record["p_binder"] = ans["noul"]
            record["y_pred"] = "binder" if ans["noul"] >= 0.5 else "non_binder"
        except Exception as exc:  # noqa: BLE001 -- surface any failure per-row, keep going
            record["y_pred"], record["p_binder"], record["error"] = None, None, str(exc)
        rows.append(record)
        time.sleep(delay)
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"  scored {i + 1}/{len(df)}")
    return df.merge(pd.DataFrame(rows), on="uuid", how="left")


def evaluate(scored_df: pd.DataFrame) -> None:
    scored = scored_df.dropna(subset=["y_pred"]).copy()
    n_failed = len(scored_df) - len(scored)
    if n_failed:
        print(f"[warn] {n_failed} requests failed and are excluded from scoring "
              f"(see the 'error' column).")
 
    y_true = (scored["y_true"] == "binder").astype(int)
    y_pred = (scored["y_pred"] == "binder").astype(int)
    majority_baseline = max(y_true.mean(), 1 - y_true.mean())
 
    print(f"\nn scored: {len(scored)}")
    print(f"class balance (true binder rate): {y_true.mean():.3f}")
    print(f"majority-class baseline accuracy: {majority_baseline:.3f}")
    print(f"Simple Jev accuracy:              {accuracy_score(y_true, y_pred):.3f}")
 
    if scored["p_binder"].notna().all() and y_true.nunique() == 2:
        print(f"Simple Jev ROC-AUC (P binder):    {roc_auc_score(y_true, scored['p_binder']):.3f}")
 
    print("\nconfusion matrix [rows=true, cols=pred], order=[non_binder, binder]:")
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred, target_names=["non_binder", "binder"]))
 
    print("per-target accuracy:")
    per_target = (
        scored.assign(correct=(y_true.values == y_pred.values))
        .groupby("target")["correct"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )
    print(per_target.to_string())