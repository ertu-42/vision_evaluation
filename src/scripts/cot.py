import json
import PIL.Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import os
import glob
from transformers import AutoTokenizer
from pydantic import BaseModel
from enum import Enum

# ---KONFIGURATION---
input_dir = "/netscratch/eilhan/data/outputs"
output_dir = "/netscratch/eilhan/results/gemma-3-12b"

# Modell wählen
model_name = "google/gemma-3-12b-it"
# model_name = "Qwen/Qwen3-VL-32B-Instruct"

# TP_SIZE = 2  # WICHTIG: 2 GPUs für große Modelle nutzen!

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ---ROLLEN & STRATEGIEN---
ROLES = [
    (
        "Neutral",
        "Du bist ein hilfreicher Assistent."
    ),
    (
        "Professor",
        "Du bist ein Mathematik-Professor und Logik-Experte. "
        "Du arbeitest formal, präzise und strukturiert."
    ),
]

STRATEGIES = [
    (
        "CoT",
        "Löse die Aufgabe logisch und gehe Schritt für Schritt vor. "
        "Nutze alle gegebenen Informationen aus Text und Bild. "
        "Antworte ausschließlich im JSON-Format gemäß Schema. "
        "Dokumentiere deinen Lösungsweg als Kurzfassung im Feld 'thoughts'. "
        "Gib im Feld 'answer' den  Lösungsbuchstaben aus {A,B,C,D,E}. "
    )
]

# ---SCHEMA FÜR STRUCTURED OUTPUT---
class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class AnswerSchema(BaseModel):
    thoughts: str
    answer: AnswerChoice


json_schema = AnswerSchema.model_json_schema()
guided_params = GuidedDecodingParams(json=json_schema)

# ---INITIALISIERUNG---
print(f">>> Lade Modell: {model_name}...")
llm = LLM(
    model=model_name,
    #tensor_parallel_size=TP_SIZE,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    limit_mm_per_prompt={"image": 6},
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Sampling Parameter
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    repetition_penalty=1.0,
    guided_decoding=guided_params
)

# Dateien suchen
json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
print(f">>> Gefunden: {len(json_files)} Dateien.")

# ---EXPERIMENT-SCHLEIFE---
for role_name, role_prompt in ROLES:
    for strat_name, strat_prompt in STRATEGIES:

        exp_id = f"{role_name}_{strat_name}"
        result_file = os.path.join(output_dir, f"{exp_id}.json")

        if os.path.exists(result_file):
            print(f"--- Überspringe {exp_id} (schon vorhanden) ---")
            continue

        print(f"\n" + "=" * 60)
        print(f"STARTE EXPERIMENT: {exp_id}")
        print("=" * 60)

        all_results = []

        # Datei-Loop
        for file_path in json_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_class = data.get("metadata", {}).get("class")
            instances = data.get("instances", [])
            if not instances:
                continue

            current_prompts = []
            current_ids = []
            current_correct = []

            # Task-Loop
            for task in instances:
                content = []
                image_inputs = []

                # Frage-Text zuerst
                content.append({"type": "text", "text": f"Löse die folgende Aufgabe: {task.get('description')}\n\n"})

                # Aufgabenbild (falls vorhanden)
                if task.get("questionImage"):
                    path = os.path.join(input_dir, task.get("questionImage"))
                    if not os.path.exists(path):
                        path = os.path.join(os.path.dirname(input_dir), task.get("questionImage"))
                    if os.path.exists(path):
                        content.append({"type": "text", "text": "Aufgabenbild:\n"})
                        content.append({"type": "image"})
                        content.append({"type": "text", "text": "\n"})
                        image_inputs.append(PIL.Image.open(path).convert("RGB"))

                # Antworten
                content.append({"type": "text", "text": "Antwortmöglichkeiten:\n"})

                for ans in task.get("answers", []):
                    label = ans.get("label", "?")

                    if ans.get("type") == "image" and ans.get("value"):
                        path = os.path.join(input_dir, ans.get("value"))
                        if not os.path.exists(path):
                            path = os.path.join(os.path.dirname(input_dir), ans.get("value"))
                        if os.path.exists(path):
                            content.append({"type": "text", "text": f"{label}: (Bild)\n"})
                            content.append({"type": "image"})
                            content.append({"type": "text", "text": "\n"})
                            image_inputs.append(PIL.Image.open(path).convert("RGB"))
                        else:
                            content.append({"type": "text", "text": f"{label}: [Bild fehlt]\n"})
                    else:
                        content.append({"type": "text", "text": f"{label}: {ans.get('value')}\n"})

                # Strategie ans Ende
                content.append({"type": "text", "text": f"\n{strat_prompt}"})

                # Chat-Template anwenden
                messages = [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": content}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                current_prompts.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": image_inputs}
                })
                current_ids.append(task.get("id"))
                current_correct.append(task.get("correct"))

            # Inferenz für diese Datei
            try:
                outputs = llm.generate(current_prompts, sampling_params=sampling_params)

                for i, output in enumerate(outputs):
                    raw = output.outputs[0].text

                    try:
                        parsed = json.loads(raw)
                        pred = str(parsed.get("answer", "")).strip()
                        thoughts = parsed.get("thoughts", "")
                    except Exception as e:
                        pred = "FEHLER"
                        thoughts = f"Parsing Error: {str(e)}\nRaw: {raw}"

                    all_results.append({
                        "experiment": exp_id,
                        "source_file": filename,
                        "class": file_class,
                        "id": current_ids[i],
                        "y_true": current_correct[i],
                        "y_pred": pred,
                        "thoughts": thoughts,
                        "role": role_name,
                        "strategy": strat_name
                    })

            except Exception as e:
                print(f"!!! Fehler bei Datei {filename}: {e}")
                continue

        # Speichern nach jedem Durchlauf
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2 ,ensure_ascii=False)

        print(f">>> Experiment {exp_id} abgeschlossen. Ergebnisse: {len(all_results)}")

print("\n>>> ALLE EXPERIMENTE FERTIG!")
