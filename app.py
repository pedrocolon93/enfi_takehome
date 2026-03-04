"""FastAPI application for LLM order dependency evaluation.

Orchestrates ModelEvaluator and DatasetEvaluator to run MCQ evaluations
with permuted answer orderings, compute metrics (accuracy, precision/recall,
ODS), and serve results via a web interface with Chart.js visualizations.
"""

import csv
import io
import json
from collections import Counter

import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model_evaluator import ModelEvaluator
from dataset_evaluator import DatasetEvaluator

app = FastAPI(title="LLM Order Dependency Evaluator")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global instances — loaded once at startup
model_evaluator = ModelEvaluator()
dataset_evaluator = DatasetEvaluator()

# Store raw results for CSV export
_last_raw_results: list[dict] = []

LABELS = ["A", "B", "C", "D", "E"]


def _settings_context() -> dict:
    """Return current model/dataset/API/generation settings for the templates."""
    gp = model_evaluator.gen_params
    return {
        "current_model": model_evaluator.model_name,
        "current_dataset": dataset_evaluator.dataset_name,
        "api_base_url": model_evaluator.api_base_url or "",
        "api_key": model_evaluator.api_key or "",
        "api_model": model_evaluator.api_model or "",
        "is_api_mode": model_evaluator.is_api_mode,
        "gen_max_new_tokens": gp.max_new_tokens,
        "gen_temperature": gp.temperature,
        "gen_top_p": gp.top_p,
        "gen_seed": gp.seed if gp.seed is not None else "",
        "gen_batch_size": gp.batch_size,
    }


@app.on_event("startup")
async def startup():
    """Load model and dataset when the server starts."""
    model_evaluator.load_model()
    dataset_evaluator.load_dataset()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main analysis page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": None,
        "all_results": None,
        **_settings_context(),
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Serve the settings page."""
    return templates.TemplateResponse("settings.html", {
        "request": request,
        **_settings_context(),
    })


@app.post("/test_api")
async def test_api(request: Request):
    """Send a 'Say hello world' test prompt to a remote API endpoint.

    Expects JSON body: {api_base_url, api_key, api_model}
    Returns JSON: {success: bool, response: str, error: str}
    """
    try:
        body = await request.json()
        base_url = body.get("api_base_url", "").strip()
        api_key = body.get("api_key", "").strip() or "not-needed"
        api_model = body.get("api_model", "").strip()

        if not base_url:
            return JSONResponse({"success": False, "response": "", "error": "API Base URL is required."})
        if not api_model:
            return JSONResponse({"success": False, "response": "", "error": "Model name is required."})

        try:
            from openai import OpenAI
        except ImportError:
            return JSONResponse({"success": False, "response": "",
                                 "error": "The 'openai' package is not installed. Run: pip install openai"})

        client = OpenAI(base_url=base_url, api_key=api_key)
        completion = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": "Say hello world."}],
            max_tokens=50,
            temperature=0.0,
        )
        content = completion.choices[0].message.content
        text = content.strip() if content else "(empty response)"
        return JSONResponse({"success": True, "response": text, "error": ""})
    except Exception as e:
        return JSONResponse({"success": False, "response": "", "error": str(e)})


@app.post("/settings", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    model_name: str = Form(""),
    dataset_name: str = Form(""),
    api_base_url: str = Form(""),
    api_key: str = Form(""),
    api_model: str = Form(""),
    gen_max_new_tokens: int = Form(10),
    gen_temperature: float = Form(0.0),
    gen_top_p: float = Form(1.0),
    gen_seed: str = Form("42"),
    gen_batch_size: int = Form(8),
):
    """Update the model, dataset, and generation parameters."""
    api_base_url = api_base_url.strip() or None
    api_key = api_key.strip() or None
    api_model = api_model.strip() or None
    model_name = model_name.strip()
    dataset_name = dataset_name.strip()

    # Determine what changed
    api_changed = (
        api_base_url != model_evaluator.api_base_url
        or api_key != model_evaluator.api_key
        or api_model != model_evaluator.api_model
    )
    model_changed = model_name != model_evaluator.model_name
    dataset_changed = dataset_name != dataset_evaluator.dataset_name

    # Apply generation parameters (no reload needed)
    gp = model_evaluator.gen_params
    gp.max_new_tokens = max(1, gen_max_new_tokens)
    gp.temperature = max(0.0, gen_temperature)
    gp.top_p = max(0.0, min(1.0, gen_top_p))
    gp.batch_size = max(1, gen_batch_size)
    seed_str = gen_seed.strip()
    gp.seed = int(seed_str) if seed_str else None

    # Apply API settings
    model_evaluator.api_base_url = api_base_url
    model_evaluator.api_key = api_key
    model_evaluator.api_model = api_model

    if model_changed or api_changed:
        if model_name:
            model_evaluator.model_name = model_name
        model_evaluator.model = None
        model_evaluator.tokenizer = None
        model_evaluator.load_model()

    if dataset_changed and dataset_name:
        dataset_evaluator.dataset_name = dataset_name
        dataset_evaluator.dataset = None
        dataset_evaluator.load_dataset()

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings_saved": True,
        **_settings_context(),
    })


@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate(
    request: Request,
    position: str = Form(...),
    num_questions: int = Form(50),
):
    """Run evaluation for a single answer position permutation.

    Args:
        position: Where to place the golden answer ("A"-"E" or "original").
        num_questions: Number of questions to sample from the dataset.
    """
    global _last_raw_results

    sample = dataset_evaluator.sample_dataset(num_questions, seed=42)
    permuted = dataset_evaluator.permute_dataset(sample, position)
    raw_results = model_evaluator.run_dataset_on_model(permuted)

    _last_raw_results = raw_results
    metrics = compute_metrics(raw_results, position)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": metrics,
        "all_results": None,
        "position": position,
        "num_questions": num_questions,
        "chart_data": json.dumps(metrics["letter_counts"]),
        **_settings_context(),
    })


@app.post("/evaluate_all", response_class=HTMLResponse)
async def evaluate_all(
    request: Request,
    num_questions: int = Form(50),
):
    """Run evaluation across all permutations (A-E + original) and compute ODS."""
    global _last_raw_results

    sample = dataset_evaluator.sample_dataset(num_questions, seed=42)

    all_metrics = {}
    all_raw = []

    for position in ["original"] + LABELS:
        permuted = dataset_evaluator.permute_dataset(sample, position)
        raw_results = model_evaluator.run_dataset_on_model(permuted)
        # Tag each result with the gold_position for CSV
        for r in raw_results:
            r["gold_position"] = position
        all_raw.extend(raw_results)
        all_metrics[position] = compute_metrics(raw_results, position)

    _last_raw_results = all_raw
    ods = compute_ods(all_metrics)
    insight = compute_bias_insight(all_metrics)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": None,
        "all_results": all_metrics,
        "ods": ods,
        "insight": insight,
        "num_questions": num_questions,
        "all_chart_data": json.dumps({
            pos: m["letter_counts"] for pos, m in all_metrics.items()
        }),
        **_settings_context(),
    })


@app.get("/export_csv")
async def export_csv():
    """Export the most recent evaluation results as a CSV file.

    CSV columns:
        question_id, question_text, gold_position, original_answer_ordering,
        permuted_answer_ordering, original_answer_key, permuted_answer_key,
        model_answer, model_raw_response, correct
    """
    if not _last_raw_results:
        return HTMLResponse("No results to export. Run an evaluation first.",
                            status_code=400)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id", "question", "gold_position", "original_ordering",
        "permuted_ordering", "original_answer_key", "permuted_answer_key",
        "model_answer", "model_raw_response", "correct",
    ])
    writer.writeheader()

    for row in _last_raw_results:
        writer.writerow({
            "id": row["id"],
            "question": row["question"],
            "gold_position": row.get("gold_position", ""),
            "original_ordering": row["original_ordering"],
            "permuted_ordering": row["permuted_ordering"],
            "original_answer_key": row["original_answer_key"],
            "permuted_answer_key": row["permuted_answer_key"],
            "model_answer": row["model_answer"],
            "model_raw_response": row["model_raw_response"],
            "correct": row["correct"],
        })

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluation_results.csv"},
    )


def compute_metrics(results: list[dict], position: str) -> dict:
    """Compute evaluation metrics from a set of results.

    Args:
        results: List of result dicts from run_dataset_on_model.
        position: The permutation position these results correspond to.

    Returns:
        Dict with keys: position, accuracy, total, correct, letter_counts,
        per_letter (precision/recall/F1 per letter), stdev.
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # Count how many times the model selected each letter
    answer_counts = Counter(r["model_answer"] for r in results)
    letter_counts = {lbl: answer_counts.get(lbl, 0) for lbl in LABELS}
    if "X" in answer_counts:
        letter_counts["X"] = answer_counts["X"]

    # Per-letter precision and recall
    per_letter = {}
    for lbl in LABELS:
        tp = sum(1 for r in results
                 if r["permuted_answer_key"] == lbl and r["model_answer"] == lbl)
        fp = sum(1 for r in results
                 if r["permuted_answer_key"] != lbl and r["model_answer"] == lbl)
        fn = sum(1 for r in results
                 if r["permuted_answer_key"] == lbl and r["model_answer"] != lbl)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        per_letter[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    counts_array = [letter_counts.get(lbl, 0) for lbl in LABELS]
    stdev = float(np.std(counts_array))

    return {
        "position": position,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "letter_counts": letter_counts,
        "per_letter": per_letter,
        "stdev": round(stdev, 4),
    }


def compute_ods(all_results: dict) -> float:
    """Compute the Option Dependency Score (ODS).

    ODS measures how much the model's answer distribution changes when
    the correct answer is moved to different positions. It is the normalized
    average variance of per-letter selection probabilities across all 5
    positional permutations (A-E).

    An ODS of 0.0 means the model is perfectly invariant to answer position.
    An ODS of 1.0 means the model's answers completely depend on position.

    Args:
        all_results: Dict mapping position -> metrics dict. Must include
                     keys "A" through "E" (the "original" key is excluded
                     from the ODS calculation).

    Returns:
        ODS value between 0.0 and 1.0.
    """
    per_letter_variances = []
    for letter in LABELS:
        fractions = []
        for perm_key in LABELS:
            metrics = all_results[perm_key]
            total = metrics["total"]
            count = metrics["letter_counts"].get(letter, 0)
            fractions.append(count / total if total > 0 else 0.0)
        per_letter_variances.append(float(np.var(fractions)))

    mean_var = float(np.mean(per_letter_variances))
    max_var = 0.16  # Theoretical max for 5 permutations
    ods = min(mean_var / max_var, 1.0)
    return round(ods, 4)


def compute_bias_insight(all_results: dict) -> dict:
    """Identify which position shows the strongest positional bias.

    For each permutation (A-E), sums the counts of all non-gold letters
    (i.e., the "other" answers). The permutation with the lowest sum of
    non-gold answers has the strongest bias toward its gold position,
    because the model is concentrating most of its selections on the
    gold letter and ignoring the rest.

    Args:
        all_results: Dict mapping position -> metrics dict.

    Returns:
        Dict with 'most_biased_position', 'non_gold_sum', and 'detail'
        containing per-position non-gold sums.
    """
    detail = {}
    for pos in LABELS:
        if pos not in all_results:
            continue
        counts = all_results[pos]["letter_counts"]
        # Sum counts for all letters OTHER than the gold position
        non_gold_sum = sum(
            counts.get(lbl, 0) for lbl in LABELS if lbl != pos
        )
        detail[pos] = {
            "non_gold_sum": non_gold_sum,
            "gold_count": counts.get(pos, 0),
            "total": all_results[pos]["total"],
        }

    if not detail:
        return {"most_biased_position": None, "detail": {}}

    most_biased = min(detail, key=lambda p: detail[p]["non_gold_sum"])

    return {
        "most_biased_position": most_biased,
        "detail": detail,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
