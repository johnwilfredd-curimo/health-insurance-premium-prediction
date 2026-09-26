# Health Insurance Premium Prediction

Predicts the annual health insurance premium to quote a customer, so an agent can give a number during
the first conversation instead of waiting on an underwriter.

**[Live demo](https://health-insurance-premium-jwc.streamlit.app/)** ·
**[API docs](https://insurance-api.srv1818955.hstgr.cloud/docs)** ·
**[Notebook](https://nbviewer.org/github/johnwilfredd-curimo/health-insurance-premium-prediction/blob/main/notebooks/health_insurance_premium_prediction.ipynb)**

---

## The problem

Shield Insurance quotes premiums manually. The statement of work set two hard numbers, and the second
one is the one that bites:

- R² above **97%**
- on **at least 95% of predictions**, the gap between predicted and actual premium must be under **10%**

A model can clear the R² bar while still failing the second. R² is an average over the whole dataset;
the SOW is a promise about **individual customers**. That distinction drove the entire project.

## Result

Reaching it took three passes through the lifecycle, not one.

| Pass | Segment | Model | Off by >10% | SOW (≤5%) |
|---|---|---|---|---|
| 1 | all ages | single model | ~30% | ❌ |
| 2A | age ≤25 | segmented | ~73% | ❌ |
| 2B | age 26+ | segmented | ~0.3% | ✅ |
| 3A | age ≤25 | segmented + `genetical_risk` | ~2% | ✅ |
| 3B | age 26+ | segmented + `genetical_risk` | ~0.3% | ✅ |

**Shipped models:**

| Artifact | Model | R² |
|---|---|---|
| `model_young.joblib` | `LinearRegression` | 0.9883 train / 0.9887 test |
| `model_rest.joblib` | `XGBRegressor`, RandomizedSearchCV | 0.9971 best CV score |

### Why three passes

**Pass 1: the average hid the failure.** The first model scored well on R² but missed the 10% band on
roughly 30% of customers. Every one of those is a real person overcharged or undercharged.

**Pass 2: error analysis found where.** Plotting residuals against features showed the errors weren't
spread evenly; they were concentrated almost entirely in customers aged 25 and under. Splitting into two
models fixed the over-25s immediately (0.3% extreme errors) and made the under-25 problem worse in
isolation: **73%**.

**Pass 3: the model asked for a feature.** A 73% failure rate on one segment isn't a tuning problem,
it's a missing-information problem: nothing in the dataset explained young customers' premiums. That
finding went back to the business as a data request, and a `genetical_risk` field came back. Retraining
the young-segment model with it dropped extreme errors from 73% to **2%**.

The third pass is the point of this project. The useful output of a failing model was **a specific,
justified data request**, not a different algorithm.

## Architecture

One model, two interfaces:

```
src/prediction.py   ← loads artifacts, builds the feature vector, predicts
   ├── streamlit_app.py   human-facing demo
   └── api/main.py        machine-facing JSON service
```

Both import the same `predict()`. There is no second copy of the preprocessing logic, so the demo and the
API cannot drift apart. The Postman suite asserts they agree.

The model is **segmented by age**: `LinearRegression` for ages 18–25, `XGBRegressor` for 26+, each with
its own fitted `MinMaxScaler`. `src/prediction.py` routes on age and picks the matching pair.

```
health-insurance-premium-prediction/
├─ streamlit_app.py          Streamlit entrypoint
├─ src/prediction.py         shared prediction module
├─ api/main.py               FastAPI service
├─ postman/                  Postman collection + run screenshot
├─ artifacts/                2 models + 2 scalers (.joblib)
├─ data/                     4 .xlsx datasets, one per iteration
├─ notebooks/                the full end-to-end notebook
└─ Dockerfile                serves the API on port 7860
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS / Linux
# Windows Git Bash:    source .venv/Scripts/activate
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The API, from the repo root:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/docs>.

```bash
curl -X POST http://127.0.0.1:8000/predict_premium \
  -H "Content-Type: application/json" \
  -d '{"age":35,"number_of_dependants":2,"income_lakhs":12,"genetical_risk":2,
       "insurance_plan":"Gold","employment_status":"Salaried","gender":"Male",
       "marital_status":"Married","bmi_category":"Normal","smoking_status":"No Smoking",
       "region":"Northwest","medical_history":"No Disease"}'
```

```json
{ "predicted_premium": 24567, "currency": "INR" }
```

To re-run the notebook: `pip install -r requirements-dev.txt`, then run it from the `notebooks/` folder.

## API tests

`postman/` holds a Postman collection covering the health check, the happy path, an ordered pair
asserting a regular smoker is quoted more than an otherwise identical non-smoker, and two validation
failures. `base_url` is a collection variable defaulting to `http://127.0.0.1:8000`, so it runs
with no setup:

```bash
npx newman run postman/health-insurance-premium.postman_collection.json
```

Against a deployed instance, override that one variable:

```bash
npx newman run postman/health-insurance-premium.postman_collection.json \
    --env-var base_url=https://<host>
```

## Attribution

Project brief, dataset and baseline approach from the Codebasics course *Master Machine Learning for Data
Science & AI* (codebasics.io). This repository is my own end-to-end rebuild: the lifecycle structure,
analysis write-up, serving layer, API, test suite and deployment are mine.

Self-directed learning project; not commissioned client work. "Shield Insurance" and "AtliQ AI" are the
course's fictional companies.
