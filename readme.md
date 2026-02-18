# Web Server Log Classifier

A machine learning model that classifies web server log entries as benign or malicious. It is trained using a RandomForestClassifier on labeled HTTP log data.

---

## Requirements

```bash
pip install scikit-learn pandas joblib
```

---

## Dataset

The model expects a CSV file named `labeled.csv` with the following columns:

```
ip, time, method, url, protocol, status, size, referrer, user_agent, extra, no, label, type
```

The `label` column is the target: `0` for benign and `1` for attack.

---

## Training the Model

Open `web_server_log_classifier.ipynb` in Jupyter and run all cells from top to bottom.

The notebook will do the following in order:

1. Load the dataset from `labeled.csv`
2. Drop unused columns (`no`, `extra`, `time`)
3. Encode categorical columns using LabelEncoder
4. Scale all features using StandardScaler
5. Split the data into 80% training and 20% test
6. Train a RandomForestClassifier
7. Print accuracy and a full classification report
8. Save the model and scaler to disk

If your machine struggles with large data, set a row limit when loading:

```python
df = pd.read_csv("labeled.csv", nrows=1000000)
```

One million rows is more than enough for a well-performing model.

---

## Saving the Model

At the end of the notebook, the model and scaler are saved using joblib:

```python
import joblib

joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

Both files will appear in the same folder as the notebook.

---

## Loading and Using the Model

To load the model in a new script or notebook without retraining:

```python
import joblib

clf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
```

To predict on a sample from the training data:

```python
clf.predict([X_train[99]])
```

To predict on a real row from the dataset:

```python
row = df[df["label"] == 0].iloc[[0]].drop(columns=["label"])
row_scaled = scaler.transform(row)
clf.predict(row_scaled)
```

---

## Output

| Value | Meaning |
|-------|---------|
| 0     | Benign  |
| 1     | Attack  |

---

## Notes

- Always save and load both `model.pkl` and `scaler.pkl` together. Using the model without the correct scaler will produce wrong predictions.
- If you retrain the model, both files will be overwritten.
- The model was trained on `labeled.csv`. If your traffic patterns differ significantly, consider retraining on your own labeled data.