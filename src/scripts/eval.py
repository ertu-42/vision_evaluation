import json
import os
from sklearn.metrics import accuracy_score

# --- KONFIGURATION ---
results_path = "/netscratch/eilhan/results/gemma-3-12b/Professor_CoT.json"

if not os.path.exists(results_path):
    print(f"Fehler: Datei {results_path} nicht gefunden.")
    exit()

print(f"Lade Ergebnisse aus: {results_path}")
with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

labels = ["A", "B", "C", "D", "E"]

# --- DATENSAMMLUNG ---
y_true_valid = []
y_pred_valid = []

total_tasks = 0
format_errors = 0
correct_all = 0

# Nach Klassenstufe gruppiert
grade_stats = {}

def init_grade(class_value: str):
    if class_value not in grade_stats:
        grade_stats[class_value] = {
            "correct_all": 0,
            "total_all": 0,
            "correct_valid": 0,
            "total_valid": 0,
            "errors": 0
        }

for r in results:
    true_label = r.get("y_true")
    if not true_label:
        continue

    total_tasks += 1

    pred = r.get("y_pred", "FEHLER")
    if pred is None:
        pred = "FEHLER"

    class_value = r["class"]
    init_grade(class_value)

    grade_stats[class_value]["total_all"] += 1

    # Accuracy (all)
    if pred == true_label:
        correct_all += 1
        grade_stats[class_value]["correct_all"] += 1

    # Valid / Error-Rate
    if pred not in labels:
        format_errors += 1
        grade_stats[class_value]["errors"] += 1
    else:
        y_true_valid.append(true_label)
        y_pred_valid.append(pred)

        grade_stats[class_value]["total_valid"] += 1
        if pred == true_label:
            grade_stats[class_value]["correct_valid"] += 1

valid_tasks = len(y_true_valid)
correct_valid = sum(1 for yt, yp in zip(y_true_valid, y_pred_valid) if yt == yp)

# --- AUSGABE ---
print("\n" + "=" * 70)
print(f"       EVALUATION ({total_tasks} Aufgaben)       ")
print("=" * 70)

acc_all = (correct_all / total_tasks) if total_tasks > 0 else 0.0
acc_valid = accuracy_score(y_true_valid, y_pred_valid) if valid_tasks > 0 else 0.0
error_rate = (format_errors / total_tasks) if total_tasks > 0 else 0.0

print(f"Total Tasks:          {total_tasks}")
print(f"Valid Tasks:          {valid_tasks}")
print(f"Format Errors:        {format_errors}")
print("-" * 70)
print(f"ACCURACY (all):       {acc_all * 100:.2f}%  ({correct_all}/{total_tasks})")
print(f"ACCURACY (valid):     {acc_valid * 100:.2f}%  ({correct_valid}/{valid_tasks})")
print(f"ERROR RATE:           {error_rate * 100:.2f}%  ({format_errors}/{total_tasks})")
print("=" * 70)

# Ergebnisse nach Klassenstufen
print("\nACCURACY NACH KLASSENSTUFEN:")
print("-" * 70)

class_order = ["3-4", "5-6", "7-8", "9-10", "11-13"]

for class_value in class_order:
    if class_value not in grade_stats:
        print(f"Klasse {class_value}: - (keine Aufgaben)")
        continue

    stats = grade_stats[class_value]
    tot_all = stats["total_all"]
    tot_valid = stats["total_valid"]

    acc_all_c = (stats["correct_all"] / tot_all) if tot_all > 0 else 0.0
    acc_valid_c = (stats["correct_valid"] / tot_valid) if tot_valid > 0 else 0.0
    error_rate_c = (stats["errors"] / tot_all) if tot_all > 0 else 0.0

    print(f"Klasse {class_value}:")
    print(f"  ACCURACY (all):       {acc_all_c * 100:.2f}%  ({stats['correct_all']}/{tot_all})")
    print(f"  ACCURACY (valid):     {acc_valid_c * 100:.2f}%  ({stats['correct_valid']}/{tot_valid})")
    print(f"  ERROR RATE:           {error_rate_c * 100:.2f}%  ({stats['errors']}/{tot_all})")

print("=" * 70)
