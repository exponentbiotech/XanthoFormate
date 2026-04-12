# Formate Biorefinery Streamlit App

This workspace now includes a local Streamlit interface for the `formate_biorefinery_model` package.

## What the app does

- lets you change biological performance, input costs, financing, and labor assumptions with sliders
- recalculates TEA and LCA outputs for the selected scenario
- shows the exact source, URL, year, confidence, and notes behind every currently used assumption
- includes a reference-centric table showing which facts were pulled from each source
- renders the existing figure suite with a description box for how each figure is calculated
- includes a Groq-powered chat panel grounded in the current scenario and figures

## Run locally

From `/Users/justin/Downloads`:

```bash
python -m pip install -r requirements.txt
export GROQ_API_KEY="your_free_groq_api_key"
streamlit run streamlit_app.py
```

If you do not set `GROQ_API_KEY`, the rest of the app still works and only the chat tab stays disabled.

## App notes

- The Streamlit app entrypoint is `streamlit_app.py`.
- Figure descriptions are stored in `formate_biorefinery_model/data/figure_metadata.json`.
- Full assumption provenance comes from the CSV-backed model inputs in `formate_biorefinery_model/data/`.
- Favorable LCA comparisons in the climate figure use renewable electricity, biogenic waste CO2, SCP carbon credit, and SCP displacement credit.

## Groq behavior

The in-app assistant is instructed to:

- stay grounded in the current app state when asked about the currently displayed scenario
- use the current model structure and assumptions to discuss alternative scenarios when asked
- avoid inventing references and instead rely on the app’s source rows and figure descriptions
