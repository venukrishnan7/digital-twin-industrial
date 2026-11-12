# Digital Twin Validation Dashboard

A Streamlit dashboard for demonstrating the existing simulated factory twin and Phase 6B–6E Random Forest/hybrid experiments.

## 1. Requirements
- Python 3.9 or newer
- Existing experiment CSVs (optional for the UI to launch; needed to inspect actual row-level outputs)

## 2. Setup on Windows PowerShell
Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## 3. Add existing results
Create a folder named `results` beside `app.py` and copy in whichever files you have:

- `factory_phase6_dataset.csv`
- `factory_phase6b_fault_predictions.csv`
- `factory_phase6c_hybrid_predictions.csv`
- `factory_phase6e_stress_predictions.csv`

Alternatively, upload CSV files using the sidebar in the app.

## 4. Important
- The dashboard does not retrain your model or fabricate row-level predictions.
- Summary metrics on the Overview and Stress Test tabs are the experiment summaries previously reported. Verify them against your archived metrics files before formal submission.
- Row-level metrics are recalculated only when the dashboard recognizes actual-label and prediction columns.
- The Live-style Twin tab is an illustrative UI scenario selector, not a connection to the Phase 5C simulator or a real PLC.
- This is a simulated research demonstration, not a safety-rated system.

## 5. Suggested viva demo
1. Show Overview and explain the architecture.
2. Select Normal operation in Live-style Twin.
3. Switch to Slow drilling, Part-sensor failure, and Clamp interlock; explain the alert logic.
4. Upload/show the saved Phase 6B and 6C prediction CSVs.
5. Show the Phase 6E stress metrics and explain why recall decreases under perturbed inputs.
6. Finish by stating the limitations: simulated data, limited scenarios, and no physical PLC validation.
