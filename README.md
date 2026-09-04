# crop-disease-risk-engine
It uses dual-pathway crop disease diagnostics combining ResNet50V2 deep learning with real-time microclimate risk modeling and prevention measures along with agronomic spray advisories via Streamlit.
# 🌾 Crop Disease Risk Engine & Diagnostic Advisory

An end-to-end ag-tech diagnostic platform combining deep learning vision models with localized microclimate risk modeling. The engine diagnoses foliar crop pathologies and projects secondary field outbreak risks to recommend targeted agronomic interventions.

---

## 📌 System Architecture

The engine operates on a **dual-pathway evaluation pipeline**:

```text
                  ┌────────────────────────┐
                  │   Uploaded Leaf Photo  │
                  └───────────┬────────────┘
                              │
                              ▼
                 [ Pathway 1: ResNet50V2 ]
             Vision-based Pathogen Classification
                              │
                              ├────────────────────────┐
                              ▼                        ▼
                   [ Diagnosis & Match % ]      [ Growth Parameters ]
                              │                        │
┌──────────────────────┐      │                        ▼
│  Real-Time Field API │ ─────┼──────────► [ Pathway 2: Climate Engine ]
│ (Temp, Hum, Precip)  │      │             Vulnerability Matrix (R_env)
└──────────────────────┘      │                        │
                              └───────────┬────────────┘
                                          │
                                          ▼
                             [ Decision Fusion Engine ]
                              Total Field Threat Index
                                          │
                                          ▼
                          [ Actionable Advisory Panel ]
                          - Chemical / Cultural Controls
                          - Latent Incubation Windows
                          - Foliar Spray Feasibility
```

1. **Pathological Vision Assessment (ResNet50V2)**: Analyzes leaf imagery using transfer learning across 38 crop disease classes, outputting pathogen identification and diagnostic match confidence.
2. **Microclimate Vulnerability Matrix**: Connects to live weather telemetry to compute temperature suitability ($I_t$), humidity severity ($I_h$), and rainfall splash dispersion ($I_r$).
3. **Decision Fusion Engine**: Synthesizes model outputs into a weighted Outbreak Propagation Risk ($R_{\text{total}}$), categorized into actionable alert states (`High Alert`, `Moderate Risk`, `Low Spread Risk`).
4. **Foliar Spray Feasibility**: Evaluates live precipitation and humidity thresholds to advise against chemical application during active wash-off windows.

---

## 🚀 Key Features

* **Real-Time Weather Ingestion**: Automated live telemetry lookup with interactive farm geocoding.
* **Dynamic Agronomic Reporting**: Action plans detailing chemical formulations, cultural practices, pathogen optimal temperature ranges, and latent incubation periods.
* **Interactive Multi-Matrix Analytics**: Integrated Plotly radar charts and factor contribution breakdowns to explain spread dynamics.
* **Responsive Visual Feedback**: Dynamic risk cards utilizing color-coded urgency borders.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Deep Learning**: TensorFlow / Keras (ResNet50V2)
* **Data Processing**: NumPy, Pillow
* **Visualizations**: Plotly Express
* **APIs**: WeatherAPI REST Services

---

## 📦 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yash25bai11046-arch/crop-disease-risk-engine.git](https://github.com/yash25bai11046-arch/crop-disease-risk-engine.git)
cd crop-disease-risk-engine
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

*(Alternatively, install packages directly: `pip install streamlit tensorflow numpy requests plotly Pillow`)*

### 3. Model Weights Setup
Place your trained `.keras` model file in the project root directory:
```text
crop-disease-risk-engine/
├── project.py
├── class_labels.json
├── crop_disease_resnet50v2.keras   <-- Place weights file here
├── requirements.txt
└── README.md
```

### 4. Run the Application
```bash
streamlit run project.py
```
---
