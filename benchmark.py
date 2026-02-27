import joblib
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# ── Load test set ────────────────────────────────────────────────────────────
print("Loading test set...", flush=True)
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()
print(f"Test set: {len(X_test):,} rows", flush=True)
print(f"Label distribution:\n{y_test.value_counts()}\n", flush=True)

# ── Models to benchmark ──────────────────────────────────────────────────────
models_info = {
    "SGDClassifier": {
        "model":  joblib.load("./models/SGDClassifier_model.pkl"),
        "scaler": joblib.load("./models/SGDClassifier_scaler.pkl"),
    },
"RandomForest": {
    "model":  joblib.load("models/RandomForestClassifier_mode.pkl"),  # missing 'l' ← fix here
    "scaler": joblib.load("models/RandomForestClassifier_scaler.pkl"),
},
    "HistGradientBoostingClassifier": {
        "model":  joblib.load("./models/HistGradientBoostingClassifier_model.pkl"),
        "scaler": joblib.load("./models/HistGradientBoostingClassifier_scaler.pkl"),
    },
}

# ── Benchmark ────────────────────────────────────────────────────────────────
results = []

for name, obj in models_info.items():
    print(f"\n{'='*55}", flush=True)
    print(f"  🔍 Evaluating {name}...", flush=True)
    print(f"{'='*55}", flush=True)

    model  = obj["model"]
    scaler = obj["scaler"]

    # Scale
    X_scaled = scaler.transform(X_test)

    # Predict + time it
    start  = time.time()
    y_pred = model.predict(X_scaled)
    elapsed = time.time() - start

    # Metrics
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"  Confusion Matrix:")
    print(f"    True  Benign : {cm[0][0]:,}  |  False Attack: {cm[0][1]:,}")
    print(f"    False Benign : {cm[1][0]:,}  |  True  Attack: {cm[1][1]:,}")

    results.append({
        "Model":          name,
        "Accuracy":       f"{acc*100:.2f}%",
        "F1 (weighted)":  f"{f1*100:.2f}%",
        "Precision":      f"{precision*100:.2f}%",
        "Recall":         f"{recall*100:.2f}%",
        "Predict Time":   f"{elapsed:.3f}s",
    })

# ── Leaderboard ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("                      🏆 LEADERBOARD")
print(f"{'='*65}")
leaderboard = pd.DataFrame(results).sort_values("F1 (weighted)", ascending=False)
print(leaderboard.to_string(index=False))
print(f"{'='*65}")