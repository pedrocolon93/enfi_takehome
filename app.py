"""FastAPI application for LLM order dependency evaluation.

Orchestrates ModelEvaluator and DatasetEvaluator to run MCQ evaluations
with permuted answer orderings, compute metrics (accuracy, precision/recall,
RStd, chi-squared), and serve results via a web interface with Chart.js
visualizations.
"""

import csv
import io
import json
import logging
import os
import uuid
from collections import Counter

import numpy as np
from scipy import stats
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model_evaluator import ModelEvaluator
from dataset_evaluator import DatasetEvaluator

# Configure logging for all modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    logger.info("Server starting up — loading model and dataset")
    model_evaluator.load_model()
    dataset_evaluator.load_dataset()
    logger.info("Startup complete")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main analysis page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": None,
        "all_results": None,
        "dataset_size": len(dataset_evaluator.dataset) if dataset_evaluator.dataset else 0,
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

        logger.info("Testing API connection: base_url=%s, model=%s", base_url, api_model)

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
        logger.info("API test successful: %r", text[:80])
        return JSONResponse({"success": True, "response": text, "error": ""})
    except Exception as e:
        logger.error("API test failed: %s", e)
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
    logger.info("Settings update requested")
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

    logger.info("Generation params updated: temp=%.2f, max_tokens=%d, top_p=%.2f, "
                "seed=%s, batch_size=%d",
                gp.temperature, gp.max_new_tokens, gp.top_p, gp.seed, gp.batch_size)

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
    uniform_sample: str = Form(""),
):
    """Run evaluation for a single answer position permutation."""
    global _last_raw_results

    uniform = bool(uniform_sample)
    logger.info("Single evaluation: position=%s, num_questions=%d, uniform=%s",
                position, num_questions, uniform)
    model_evaluator.cancel_requested = False

    sample = dataset_evaluator.sample_dataset(num_questions, seed=42, uniform=uniform)
    permuted = dataset_evaluator.permute_dataset(sample, position)
    raw_results = model_evaluator.run_dataset_on_model(permuted)

    _last_raw_results = raw_results
    metrics = compute_metrics(raw_results, position)
    chi2 = compute_chi_squared(metrics)
    rstd = compute_rstd(metrics)
    incorrect = [r for r in raw_results if not r["correct"]]
    logger.info("Single evaluation complete: accuracy=%.2f%%, RStd=%.4f, chi2=%.4f (p=%.6f)",
                metrics["accuracy"] * 100, rstd, chi2["chi2_stat"], chi2["p_value"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": metrics,
        "all_results": None,
        "position": position,
        "num_questions": num_questions,
        "chart_data": json.dumps(metrics["letter_counts"]),
        "chi_squared": chi2,
        "rstd": rstd,
        "raw_results": raw_results,
        "incorrect_questions": incorrect,
        "uniform_sample": uniform,
        "dataset_size": len(dataset_evaluator.dataset) if dataset_evaluator.dataset else 0,
        **_settings_context(),
    })


@app.post("/evaluate_all", response_class=HTMLResponse)
async def evaluate_all(
    request: Request,
    num_questions: int = Form(50),
    uniform_sample: str = Form(""),
):
    """Run evaluation across all permutations (A-E + original) and compute RStd."""
    global _last_raw_results

    uniform = bool(uniform_sample)
    logger.info("Full evaluation: num_questions=%d, uniform=%s, running 6 permutations",
                num_questions, uniform)
    model_evaluator.cancel_requested = False

    sample = dataset_evaluator.sample_dataset(num_questions, seed=42, uniform=uniform)

    all_metrics = {}
    all_raw = []

    for position in ["original"] + LABELS:
        if model_evaluator.cancel_requested:
            logger.warning("Full evaluation cancelled during position %s", position)
            break
            
        permuted = dataset_evaluator.permute_dataset(sample, position)
        raw_results = model_evaluator.run_dataset_on_model(permuted)
        # Tag each result with the gold_position for CSV
        for r in raw_results:
            r["gold_position"] = position
        all_raw.extend(raw_results)
        all_metrics[position] = compute_metrics(raw_results, position)

    _last_raw_results = all_raw
    insight = compute_bias_insight(all_metrics)

    # RStd per permutation (primary bias metric)
    all_rstd = {pos: compute_rstd(m) for pos, m in all_metrics.items()}

    # Overall RStd: average across all permutations
    rstd_values = [v for v in all_rstd.values() if v is not None]
    overall_rstd = round(float(np.mean(rstd_values)), 4) if rstd_values else None

    # Chi-squared only on the original (unpermuted) distribution
    original_chi2 = None
    if "original" in all_metrics:
        original_chi2 = compute_chi_squared(all_metrics["original"])

    # Build incorrect questions map: question_id -> {pos: result_dict}
    incorrect_map = {}
    for r in all_raw:
        if not r["correct"]:
            qid = r["id"]
            if qid not in incorrect_map:
                incorrect_map[qid] = {
                    "question": r["question"],
                    "original_ordering": r["original_ordering"],
                    "positions": {},
                }
            incorrect_map[qid]["positions"][r.get("gold_position", "?")] = {
                "model_answer": r["model_answer"],
                "expected": r["permuted_answer_key"],
                "permuted_ordering": r["permuted_ordering"],
                "raw_response": r["model_raw_response"],
            }

    logger.info("Full evaluation complete: overall_RStd=%.4f, most_biased=%s",
                overall_rstd or 0, insight.get("most_biased_position"))
    for pos, m in all_metrics.items():
        logger.info("  %s: accuracy=%.1f%%, RStd=%.4f",
                     pos, m["accuracy"] * 100, all_rstd[pos])
    if original_chi2:
        logger.info("  Chi-squared (original): chi2=%.4f, p=%.6f, significant=%s",
                     original_chi2["chi2_stat"], original_chi2["p_value"],
                     original_chi2["significant"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": None,
        "all_results": all_metrics,
        "overall_rstd": overall_rstd,
        "insight": insight,
        "num_questions": num_questions,
        "all_chart_data": json.dumps({
            pos: m["letter_counts"] for pos, m in all_metrics.items()
        }),
        "original_chi2": original_chi2,
        "all_rstd": all_rstd,
        "raw_results": all_raw,
        "incorrect_map": incorrect_map,
        "uniform_sample": uniform,
        "dataset_size": len(dataset_evaluator.dataset) if dataset_evaluator.dataset else 0,
        **_settings_context(),
    })


@app.post("/cancel")
async def cancel_evaluation(request: Request):
    """Cancel an ongoing evaluation."""
    logger.info("Cancellation requested by user")
    model_evaluator.cancel_requested = True
    return JSONResponse({"success": True, "message": "Cancellation requested"})


@app.get("/export_csv")
async def export_csv():
    """Export the most recent evaluation results as a CSV file.

    CSV columns:
        question_id, question_text, gold_position, original_answer_ordering,
        permuted_answer_ordering, original_answer_key, permuted_answer_key,
        model_answer, model_raw_response, correct
    """
    if not _last_raw_results:
        logger.warning("CSV export requested but no results available")
        return HTMLResponse("No results to export. Run an evaluation first.",
                            status_code=400)
    logger.info("Exporting CSV: %d rows", len(_last_raw_results))

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


def compute_chi_squared(metrics: dict) -> dict:
    """Run a chi-squared goodness-of-fit test on the model's answer distribution.

    Tests whether the model's letter selections differ significantly from a
    uniform distribution (each letter equally likely = total/5).

    Args:
        metrics: Metrics dict from compute_metrics (needs letter_counts, total).

    Returns:
        Dict with chi2_stat, p_value, significant (bool at alpha=0.05),
        observed, expected.
    """
    observed = [metrics["letter_counts"].get(lbl, 0) for lbl in LABELS]
    total = sum(observed)
    if total == 0:
        return {"chi2_stat": 0.0, "p_value": 1.0, "significant": False,
                "observed": observed, "expected": [0] * 5}

    expected = [total / len(LABELS)] * len(LABELS)
    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)

    return {
        "chi2_stat": round(float(chi2_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "observed": observed,
        "expected": [round(e, 1) for e in expected],
    }


def compute_rstd(metrics: dict) -> float:
    """Compute Recall Standard Deviation (RStd).

    RStd is the standard deviation of per-letter recall values.
    RStd = 0 means the model recalls each letter equally well (unbiased).
    Higher values indicate stronger selection bias.

    Args:
        metrics: Metrics dict from compute_metrics (needs per_letter).

    Returns:
        RStd value (float).
    """
    recalls = [metrics["per_letter"][lbl]["recall"] for lbl in LABELS]
    return round(float(np.std(recalls)), 4)


# --- Custom Dataset Storage ---
CUSTOM_DATASETS_DIR = "custom_datasets"
os.makedirs(CUSTOM_DATASETS_DIR, exist_ok=True)

# Store generated datasets in memory for the current session
_generated_datasets: dict[str, list[dict]] = {}


def _load_custom_datasets_index() -> list[dict]:
    """List all saved custom datasets from disk."""
    datasets = []
    for fname in sorted(os.listdir(CUSTOM_DATASETS_DIR)):
        if fname.endswith(".json"):
            fpath = os.path.join(CUSTOM_DATASETS_DIR, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                datasets.append({
                    "filename": fname,
                    "name": data.get("name", fname),
                    "num_questions": len(data.get("questions", [])),
                    "created": data.get("created", ""),
                })
            except Exception:
                pass
    return datasets


@app.get("/generate_dataset", response_class=HTMLResponse)
async def generate_dataset_page(request: Request):
    """Serve the dataset generation page."""
    saved_datasets = _load_custom_datasets_index()
    return templates.TemplateResponse("generate_dataset.html", {
        "request": request,
        "saved_datasets": saved_datasets,
        **_settings_context(),
    })


@app.post("/generate_dataset", response_class=HTMLResponse)
async def generate_dataset(
    request: Request,
    topic: str = Form("general knowledge"),
    num_questions: int = Form(10),
    dataset_name: str = Form(""),
):
    """Generate a MCQ dataset using the configured LLM."""
    logger.info("Generating dataset: topic=%s, num_questions=%d", topic, num_questions)

    if not model_evaluator.is_api_mode and model_evaluator.model is None:
        saved_datasets = _load_custom_datasets_index()
        return templates.TemplateResponse("generate_dataset.html", {
            "request": request,
            "error": "No model loaded. Configure a model in Settings first.",
            "saved_datasets": saved_datasets,
            **_settings_context(),
        })

    prompt = (
        f"Generate exactly {num_questions} multiple-choice questions about {topic}. "
        "Each question must have exactly 5 answer options (A through E) with exactly one correct answer.\n\n"
        "Return ONLY valid JSON in this exact format (no other text):\n"
        '[\n'
        '  {\n'
        '    "question": "What is the capital of France?",\n'
        '    "choices": {"label": ["A","B","C","D","E"], '
        '"text": ["Paris","London","Berlin","Madrid","Rome"]},\n'
        '    "answerKey": "A"\n'
        '  }\n'
        ']\n'
    )

    try:
        backend = "API" if model_evaluator.is_api_mode else "local"
        logger.info("Sending generation prompt to %s backend (max_new_tokens=4096)", backend)

        if model_evaluator.is_api_mode:
            old_max = model_evaluator.gen_params.max_new_tokens
            model_evaluator.gen_params.max_new_tokens = 4096
            raw = model_evaluator._run_api_single(prompt)
            model_evaluator.gen_params.max_new_tokens = old_max
        else:
            old_max = model_evaluator.gen_params.max_new_tokens
            model_evaluator.gen_params.max_new_tokens = 4096
            raw = model_evaluator._run_local_batch([prompt])[0]
            model_evaluator.gen_params.max_new_tokens = old_max

        logger.info("Received raw response: %d chars", len(raw))
        logger.debug("Raw response preview: %r", raw[:200])

        # Try to extract JSON from the response
        json_match = raw
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            json_match = raw[start:end + 1]
            logger.info("Extracted JSON array from position %d to %d", start, end + 1)
        else:
            logger.warning("No JSON array delimiters found in response")

        questions = json.loads(json_match)
        logger.info("Parsed %d raw question objects from JSON", len(questions))

        # Validate and add IDs
        validated = []
        skipped = 0
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                logger.debug("Skipping item %d: not a dict", i)
                skipped += 1
                continue
            if "question" not in q or "choices" not in q or "answerKey" not in q:
                logger.debug("Skipping item %d: missing required fields", i)
                skipped += 1
                continue
            choices = q["choices"]
            if not isinstance(choices, dict) or "label" not in choices or "text" not in choices:
                logger.debug("Skipping item %d: invalid choices structure", i)
                skipped += 1
                continue
            if len(choices["label"]) != 5 or len(choices["text"]) != 5:
                logger.debug("Skipping item %d: expected 5 choices, got %d labels / %d texts",
                             i, len(choices["label"]), len(choices["text"]))
                skipped += 1
                continue
            if q["answerKey"] not in choices["label"]:
                logger.debug("Skipping item %d: answerKey '%s' not in labels", i, q["answerKey"])
                skipped += 1
                continue
            q["id"] = f"gen_{uuid.uuid4().hex[:8]}"
            q["question_concept"] = topic
            validated.append(q)

        logger.info("Validation complete: %d valid, %d skipped out of %d parsed",
                     len(validated), skipped, len(questions))

        if not validated:
            raise ValueError("No valid questions could be parsed from the model's response.")

        # Store in memory
        ds_id = uuid.uuid4().hex[:12]
        _generated_datasets[ds_id] = validated

        logger.info("Generated %d valid questions (requested %d)", len(validated), num_questions)

        saved_datasets = _load_custom_datasets_index()
        return templates.TemplateResponse("generate_dataset.html", {
            "request": request,
            "generated": validated,
            "generated_id": ds_id,
            "gen_topic": topic,
            "gen_name": dataset_name or f"{topic} ({len(validated)} questions)",
            "saved_datasets": saved_datasets,
            **_settings_context(),
        })

    except Exception as e:
        logger.error("Dataset generation failed: %s", e)
        saved_datasets = _load_custom_datasets_index()
        return templates.TemplateResponse("generate_dataset.html", {
            "request": request,
            "error": f"Generation failed: {e}",
            "raw_response": raw if 'raw' in dir() else "",
            "saved_datasets": saved_datasets,
            **_settings_context(),
        })


@app.post("/save_generated_dataset")
async def save_generated_dataset(
    request: Request,
    dataset_id: str = Form(""),
    dataset_name: str = Form(""),
):
    """Save a generated dataset to disk as JSON."""
    if dataset_id not in _generated_datasets:
        return JSONResponse({"success": False, "error": "Dataset not found in memory. Generate it again."})

    questions = _generated_datasets[dataset_id]
    name = dataset_name.strip() or f"generated_{dataset_id}"
    filename = f"{dataset_id}.json"
    filepath = os.path.join(CUSTOM_DATASETS_DIR, filename)

    from datetime import datetime
    data = {
        "name": name,
        "created": datetime.now().isoformat(),
        "num_questions": len(questions),
        "questions": questions,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved generated dataset: %s (%d questions)", filepath, len(questions))
    return JSONResponse({"success": True, "filename": filename, "name": name})


@app.get("/export_dataset/{dataset_id}")
async def export_dataset(dataset_id: str):
    """Export a generated/saved dataset as downloadable JSON."""
    # Check memory first
    if dataset_id in _generated_datasets:
        questions = _generated_datasets[dataset_id]
        data = {"name": f"generated_{dataset_id}", "questions": questions}
    else:
        # Check disk
        filepath = os.path.join(CUSTOM_DATASETS_DIR, f"{dataset_id}.json")
        if not os.path.exists(filepath):
            return HTMLResponse("Dataset not found.", status_code=404)
        with open(filepath) as f:
            data = json.load(f)

    output = json.dumps(data, indent=2)
    return StreamingResponse(
        iter([output]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={dataset_id}.json"},
    )


@app.post("/import_dataset", response_class=HTMLResponse)
async def import_dataset(request: Request):
    """Import a dataset from uploaded JSON file."""
    form = await request.form()
    upload = form.get("dataset_file")
    if not upload:
        return templates.TemplateResponse("settings.html", {
            "request": request,
            "import_error": "No file uploaded.",
            **_settings_context(),
        })

    try:
        content = await upload.read()
        data = json.loads(content)
        questions = data.get("questions", data if isinstance(data, list) else [])

        if not questions:
            raise ValueError("No questions found in the uploaded file.")

        # Validate
        for q in questions:
            if "question" not in q or "choices" not in q or "answerKey" not in q:
                raise ValueError(f"Invalid question format: missing required fields")

        # Save to custom_datasets
        ds_id = uuid.uuid4().hex[:12]
        name = data.get("name", upload.filename or f"imported_{ds_id}")
        filepath = os.path.join(CUSTOM_DATASETS_DIR, f"{ds_id}.json")

        from datetime import datetime
        save_data = {
            "name": name,
            "created": datetime.now().isoformat(),
            "num_questions": len(questions),
            "questions": questions,
        }
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

        logger.info("Imported dataset: %s (%d questions)", name, len(questions))

        return templates.TemplateResponse("settings.html", {
            "request": request,
            "import_success": f"Imported '{name}' with {len(questions)} questions.",
            **_settings_context(),
        })
    except Exception as e:
        logger.error("Dataset import failed: %s", e)
        return templates.TemplateResponse("settings.html", {
            "request": request,
            "import_error": f"Import failed: {e}",
            **_settings_context(),
        })


@app.post("/use_custom_dataset")
async def use_custom_dataset(request: Request, dataset_id: str = Form("")):
    """Switch to using a custom dataset for evaluation."""
    filepath = os.path.join(CUSTOM_DATASETS_DIR, f"{dataset_id}.json")
    if not os.path.exists(filepath):
        return JSONResponse({"success": False, "error": "Dataset not found."})

    with open(filepath) as f:
        data = json.load(f)

    questions = data.get("questions", [])

    # Monkey-patch the dataset_evaluator to use custom data
    class CustomDatasetWrapper:
        def __init__(self, items):
            self._items = items
        def __len__(self):
            return len(self._items)
        def __getitem__(self, idx):
            return self._items[idx]

    dataset_evaluator.dataset = CustomDatasetWrapper(questions)
    dataset_evaluator.dataset_name = data.get("name", f"custom_{dataset_id}")
    logger.info("Switched to custom dataset: %s (%d questions)",
                dataset_evaluator.dataset_name, len(questions))

    return JSONResponse({"success": True, "name": dataset_evaluator.dataset_name,
                         "num_questions": len(questions)})


if __name__ == "__main__":
    import uvicorn
    print("Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Server started at http://0.0.0.0:8000")
