# Vision Evaluation

> **Der Einfluss von Prompt-Spezifität auf
die Lösungsgenauigkeit von Vision-LLMs
bei bildbasierten Aufgaben**

Dieses Repository enthält Code und Daten zur systematischen Evaluation von Vision-Language-Modellen (VLMs) auf bildbasierten Multiple-Choice-Aufgaben des Wettbewerbs *Känguru der Mathematik*. Untersucht wird der Einfluss von Prompt-Spezifität in einem 2×2-Design aus **Rolleninstruktion** (Neutral vs. Professor) und **Prompting-Strategie** (Zero-Shot vs. Zero-Shot-CoT).

---

## Inhaltsverzeichnis

- [Überblick](#überblick)
- [Repository-Struktur](#repository-struktur)
- [Voraussetzungen](#voraussetzungen)
- [Setup](#setup)
- [Datenformat](#datenformat)
- [Promptbedingungen](#promptbedingungen)
- [Modelle](#modelle)
- [Inferenz](#inferenz)
- [Evaluation](#evaluation)
- [Reproduzierbarkeit](#reproduzierbarkeit)

---

## Überblick

| Dimension            | Ausprägung                                        |
|----------------------|---------------------------------------------------|
| **Aufgabentyp**      | Bildbasierte Multiple-Choice (A–E)                |
| **Datensatz**        | Känguru der Mathematik (2023, 2024, 2025)         |
| **Klassenstufen**    | 3–4, 5–6, 7–8, 9–10, 11–13                       |
| **Modellfamilien**   | Qwen2.5-VL, Qwen3-VL, Gemma 3                    |
| **Experimentdesign** | 2×2 (Rolle × Strategie)                           |
| **Metriken**         | Total-Accuracy · Valid-Accuracy · Error-Rate       |

## Repository-Struktur

```
vision_evaluation/
├── data/
│   ├── manual/
│   │   ├── inputs/             # Quell-PDFs (Känguru 2023–2025, je 5 Klassenstufen)
│   │   ├── outputs/            # Manuell annotierte JSON-Aufgabendateien
│   │   └── pictures/           # Aufgaben- & Antwortbilder (nach Jahr/Klassenstufe)
│   │       ├── 2023/
│   │       ├── 2024/
│   │       └── 2025/
│   └── results/                # Modelloutputs (4 JSON-Dateien pro Modell)
│       ├── gemma-3-4b/
│       ├── gemma-3-12b/
│       ├── gemma-3-27b/
│       ├── qwen2.5-vl-3b/
│       ├── qwen2.5-vl-7b/
│       ├── qwen2.5-vl-32b/
│       ├── qwen2.5-vl-72b/
│       ├── qwen3-vl-4b/
│       ├── qwen3-vl-8b/
│       └── qwen3-vl-32b/
├── src/
│   └── scripts/
│       ├── zeroshot.py         # Inferenz für Zero-Shot (P1/P2)
│       ├── cot.py              # Inferenz für Zero-Shot-CoT (P3/P4)
│       └── eval.py             # Auswertung (Accuracy, Error-Rate, nach Klassenstufe)
├── requirements.txt
└── .gitignore
```

## Voraussetzungen

- **Python** ≥ 3.10
- **CUDA**-fähige GPU (für die Inferenz mit vLLM / PyTorch)
- Passende CUDA-/cuDNN-Installation (kompatibel mit der verwendeten PyTorch-Version)

## Setup

```bash
# 1) Repository klonen
git clone https://github.com/ertu-42/vision_evaluation.git
cd vision_evaluation

# 2) Virtuelle Umgebung erstellen & aktivieren
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3) Abhängigkeiten installieren
pip install -r requirements.txt
```

> ⚠️ `vllm` und `torch` sind CUDA-/System-abhängig. Stelle sicher, dass eine kompatible CUDA-Installation vorhanden ist, bevor du `pip install` ausführst.

### Abhängigkeiten

| Paket           | Verwendung                          |
|-----------------|-------------------------------------|
| `vllm`          | Inferenz-Engine (Guided Decoding)   |
| `transformers`  | Tokenizer / Chat-Templates         |
| `pydantic`      | JSON-Schema für strukturierte Ausgabe |
| `pillow`        | Bildverarbeitung                    |
| `tqdm`          | Fortschrittsanzeige                 |
| `scikit-learn`  | Accuracy-Berechnung (`eval.py`)     |

## Datenformat

Die Quell-PDFs der Känguru-Aufgaben liegen in `data/manual/inputs/`. Die daraus manuell erstellten Aufgaben-JSON-Dateien befinden sich in `data/manual/outputs/`.

Jede JSON-Datei enthält Metadaten sowie eine Liste von Instanzen:

| Feld                         | Beschreibung                                          |
|------------------------------|-------------------------------------------------------|
| `metadata.class`             | Klassenstufe (z. B. `"3-4"`, `"5-6"`)               |
| `metadata.date`              | Wettbewerbsdatum                                      |
| `metadata.requiredTime`      | Bearbeitungszeit                                      |
| `instances[].id`             | Aufgabenkennung (z. B. `"A1"`, `"B5"`)               |
| `instances[].description`    | Aufgabentext                                          |
| `instances[].questionImage`  | Optional: Pfad zum Aufgabenbild (relativ)             |
| `instances[].answers[]`      | Antwortoptionen (`label`, `type`: text/image, `value`)|
| `instances[].correct`        | Korrekte Antwort (`"A"`–`"E"`)                       |

<details>
<summary>Beispiel-JSON (vereinfacht)</summary>

```json
{
  "metadata": {
    "filename": "kaenguru2023_34.pdf",
    "class": "3-4",
    "date": "Donnerstag, 16. März 2023",
    "requiredTime": "75 Minuten"
  },
  "instances": [
    {
      "id": "A1",
      "description": "Jonathan hat fünf gleiche Kerzen gleichzeitig angezündet …",
      "questionImage": "../pictures/2023/2023_34/pic_1.png",
      "answers": [
        { "label": "A", "type": "text", "value": "A" },
        { "label": "B", "type": "text", "value": "B" }
      ],
      "correct": "D"
    }
  ]
}
```

</details>

## Promptbedingungen

Es werden vier Promptvarianten in einem 2×2-Design verwendet:

|                     | **Zero-Shot**          | **Zero-Shot-CoT**      |
|---------------------|------------------------|------------------------|
| **Neutral**         | P1                     | P3                     |
| **Professor**       | P2                     | P4                     |

- **Neutral**: `"Du bist ein hilfreicher Assistent."`
- **Professor**: `"Du bist ein Mathematik-Professor und Logik-Experte. Du arbeitest formal, präzise und strukturiert."`
- **Zero-Shot**: Direkte Antwort im JSON-Format, nur `answer`-Feld.
- **Zero-Shot-CoT**: Schritt-für-Schritt-Lösung, `thoughts`- und `answer`-Feld.

Die Ausgabe wird über ein Pydantic-basiertes JSON-Schema via Guided Decoding (vLLM) erzwungen.

## Modelle

| Modellfamilie   | Parametergrößen          |
|-----------------|--------------------------|
| Qwen2.5-VL     | 3B, 7B, 32B, 72B        |
| Qwen3-VL       | 4B, 8B, 32B             |
| Gemma 3         | 4B, 12B, 27B            |

Ergebnisse liegen jeweils in `data/results/<modellname>/` mit vier Dateien pro Modell: `Neutral_ZeroShot.json`, `Professor_ZeroShot.json`, `Neutral_CoT.json`, `Professor_CoT.json`.

## Inferenz

In den Skripten müssen vor der Ausführung `input_dir`, `output_dir` und `model_name` angepasst werden.

### Zero-Shot (P1 / P2)

```bash
python src/scripts/zeroshot.py
```

### Zero-Shot-CoT (P3 / P4)

```bash
python src/scripts/cot.py
```

Beide Skripte lesen die annotierten JSON-Aufgaben aus `data/manual/outputs/`, laden die zugehörigen Bilder aus `data/manual/pictures/` und schreiben Ergebnisdateien nach `data/results/`. Jede Ergebnisdatei ist eine JSON-Liste mit folgenden Feldern:

| Feld          | Beschreibung                                   |
|---------------|-------------------------------------------------|
| `experiment`  | Experimentkennung (z. B. `"Neutral_ZeroShot"`)  |
| `source_file` | Quelldatei                                      |
| `class`       | Klassenstufe                                    |
| `id`          | Aufgabenkennung                                 |
| `y_true`      | Korrekte Antwort                                |
| `y_pred`      | Vorhergesagte Antwort (oder `"FEHLER"`)         |
| `role`        | Verwendete Rolleninstruktion                    |
| `strategy`    | Verwendete Prompting-Strategie                  |
| `thoughts`    | Gedankenkette (nur im CoT-Setting)              |

## Evaluation

```bash
python src/scripts/eval.py
```

> In `eval.py` muss `results_path` auf die gewünschte Ergebnisdatei in `data/results/` gesetzt werden.

Das Skript berechnet folgende Metriken (gesamt und nach Klassenstufe):

| Metrik              | Beschreibung                                                         |
|---------------------|----------------------------------------------------------------------|
| **Total-Accuracy**  | Accuracy über alle Aufgaben (ungültige Outputs zählen als falsch)    |
| **Valid-Accuracy**  | Accuracy nur über gültige Outputs (Antwort ∈ {A, B, C, D, E})       |
| **Error-Rate**      | Anteil ungültiger Outputs                                            |

## Reproduzierbarkeit

- **Deterministische Generierung**: Temperature = 0
- **Strukturierte Ausgabe**: Pydantic-Schema via Guided Decoding (vLLM), um die Auswertbarkeit zu maximieren
- **Einflussfaktoren**: Ergebnisse hängen von Modellversion, vLLM-/Transformers-Version, GPU-Setup und Tokenizer/Chat-Template ab
