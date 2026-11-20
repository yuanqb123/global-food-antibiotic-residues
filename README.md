# Antibiotic residues in global food supplies: overlooked potential for resistance selection

## Contents

- `run_analysis.py`: entry point for the main and supplementary analyses.
- `analysis/`: statistical and plotting functions.
- `config.json`: model configurations and risk parameters.
- `figure1_reported_counts.json`: counts and labels transcribed from manuscript Figure 1 for plotting.
- `requirements.txt`: dependency versions.
- `data/`: frozen analytical inputs, variable definitions and source URLs.
- `source_data/`: numerical data underlying the figures; `figure_data_index.csv` maps figures to their input files.

## Run

Python 3.11:

```bash
python -m pip install -r requirements.txt
python run_analysis.py
```

Results and figures are written to `outputs/`.

To redraw the figures from the supplied source data:

```bash
python run_analysis.py --step figures
```

Use `--help` to list individual analysis steps. Empty CSV fields indicate missing values. Response and SHAP bootstrap intervals are calculated conditional on fitted held-out models. The `.npz` source files contain numerical response-profile arrays (grid, predictions, weights and fold).
