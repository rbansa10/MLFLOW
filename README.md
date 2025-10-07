# MLFLOW
This repo has demonstartion of performing experiment tracking and other actions
Got it 👍
Here’s a **concise but detailed MLflow Cheatsheet** you can use as a quick reference for running ML experiments, tracking metrics, saving models, and managing runs.

---

# 🧾 MLflow Cheatsheet

## 🔹 1. Install & Setup

```bash
pip install mlflow
```

Start MLflow UI:

```bash
mlflow ui
```

(Default at: `http://127.0.0.1:5000`)

---

## 🔹 2. Basic Experiment Tracking

```python
import mlflow

# Start run (creates a unique run_id)
with mlflow.start_run(run_name="experiment_1"):
    mlflow.log_param("learning_rate", 0.01)   # log hyperparameter
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.85)       # log metric
    mlflow.log_metric("loss", 0.35)

    # Save artifacts (any file: plots, config, etc.)
    with open("output.txt", "w") as f:
        f.write("Model training complete.")
    mlflow.log_artifact("output.txt")
```

---

## 🔹 3. Tracking Models

```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Log model
mlflow.sklearn.log_model(model, "random_forest_model")
```

---

## 🔹 4. Nested Runs (For Complex Pipelines)

```python
with mlflow.start_run(run_name="parent_run"):
    mlflow.log_param("parent_param", "value")

    # Nested run for data preprocessing
    with mlflow.start_run(run_name="preprocessing", nested=True):
        mlflow.log_param("scaler", "standard")
```

---

## 🔹 5. Viewing Runs

* Run UI: `mlflow ui` → [http://localhost:5000](http://localhost:5000)
* Each run stores:

  * **Parameters** (`params`)
  * **Metrics** (`metrics`)
  * **Artifacts** (plots, configs, model files)
  * **Models** (saved ML model)

---

## 🔹 6. Comparing Experiments

```bash
mlflow ui
```

👉 Go to **Experiments tab** → Select multiple runs → Compare metrics visually.

---

## 🔹 7. Model Registry

Register best model:

```python
mlflow.register_model(
    "runs:/<RUN_ID>/random_forest_model",
    "RandomForestClassifier"
)
```

Promote model stage:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="RandomForestClassifier",
    version=1,
    stage="Production"
)
```

---

## 🔹 8. Serving Models

Serve model as REST API:

```bash
mlflow models serve -m runs:/<RUN_ID>/random_forest_model -p 1234
```

Send request:

```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" \
    --data '{"columns":["f1","f2"],"data":[[0.1,0.2]]}' \
    http://127.0.0.1:1234/invocations
```

---

## 🔹 9. Logging Plots

```python
import matplotlib.pyplot as plt

plt.plot([0,1,2],[0.2,0.5,0.8])
plt.savefig("plot.png")

mlflow.log_artifact("plot.png")
```

---

## 🔹 10. Useful Commands

* List runs:

```bash
mlflow experiments list
mlflow runs list --experiment-id <ID>
```

* Delete run:

```bash
mlflow gc --backend-store-uri sqlite:///mlflow.db
```

---
![Uploading image.png…]()



<img width="800" height="1000" alt="image" src="https://github.com/user-attachments/assets/858fe96f-3081-427b-91e2-dcc165b3893a" />


✅ **Workflow Summary**

1. Define experiment → `mlflow.set_experiment("exp_name")`
2. Start run → `mlflow.start_run()`
3. Log params, metrics, artifacts, models
4. Track & compare runs in UI
5. Register best model in Model Registry
6. Deploy/Serve model as API


