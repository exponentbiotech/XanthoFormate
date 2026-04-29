# Formate Biorefinery Streamlit App

This workspace now includes a local Streamlit interface for the `formate_biorefinery_model` package.

## What the app does

- lets you change biological performance, input costs, financing, and labor assumptions with sliders
- recalculates TEA and LCA outputs for the selected scenario
- shows the exact source, URL, year, confidence, and notes behind every currently used assumption
- includes a reference-centric table showing which facts were pulled from each source
- renders the existing figure suite with a description box for how each figure is calculated
- includes a deterministic model interpreter grounded in the Python TEA/LCA outputs

## Run locally

From `/Users/justin/Downloads`:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The model interpreter does not require an API key or external LLM service.

## App notes

- The Streamlit app entrypoint is `streamlit_app.py`.
- Figure descriptions are stored in `formate_biorefinery_model/data/figure_metadata.json`.
- Full assumption provenance comes from the CSV-backed model inputs in `formate_biorefinery_model/data/`.
- Favorable LCA comparisons in the climate figure use renewable electricity, biogenic waste CO2, SCP carbon credit, and SCP displacement credit.

## Model interpreter behavior

The in-app assistant:

- answers from Python model outputs, not from a free-form LLM
- compares the active scenario against recovery-method, feedstock, electricity, capacity, and LCA-credit sweeps
- reports NPV, LCOX, and GWP values directly from the same model state used by the dashboard
