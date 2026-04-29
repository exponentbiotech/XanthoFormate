# Formate Biorefinery Streamlit App

This workspace includes a local Streamlit interface for the
`formate_biorefinery_model` package.

## What the app does

- lets you change biological performance, input costs, financing, and labor assumptions with sliders
- recalculates TEA and LCA outputs for the selected scenario
- shows the exact source, URL, year, confidence, and notes behind every currently used assumption
- includes a reference-centric table showing which facts were pulled from each source
- renders the existing figure suite with a description box for how each figure is calculated
- provides a Groq-hosted assistant grounded in the live model numbers AND the underlying Python source code

## Run locally

From `/Users/justin/Downloads`:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

To enable the LLM-backed assistant, either:

- export `GROQ_API_KEY=...` in your shell, or
- set `GROQ_API_KEY` in `.streamlit/secrets.toml`, or
- paste the key into the sidebar input on first load.

If no key is set the chat falls back to a deterministic Python interpreter so it
still works — just with shorter, more templated answers.

## App notes

- The Streamlit app entrypoint is `streamlit_app.py`.
- Figure descriptions are stored in `formate_biorefinery_model/data/figure_metadata.json`.
- Full assumption provenance comes from the CSV-backed model inputs in `formate_biorefinery_model/data/`.
- Favorable LCA comparisons in the climate figure use renewable electricity, biogenic waste CO2, SCP carbon credit, and SCP displacement credit.

## Assistant behavior

The in-app chat assistant:

- pulls every numeric value verbatim from the locked Python model snapshot
  (active scenario plus comparison grids over recovery methods, feedstocks,
  capacities, electricity cases, and LCA settings)
- can answer methodology questions like "how is NPV computed?" or "where is
  the biogenic carbon credit defined?" by referring to `tea.py` and `lca.py`,
  which are sent to the LLM as part of every prompt
- handles the direction of the question correctly (best vs. worst, lowest
  vs. highest)
- falls back to a deterministic Python interpreter when no LLM key is
  configured or the LLM call fails, so the chat never breaks
