"""Run the manuscript analyses or redraw figures from generated results."""

import argparse, importlib, os, time, shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from threadpoolctl import threadpool_limits
from analysis.common import INPUTS

GROUPS = {
    "summaries": ["occurrence", "spatial_temporal_summaries"],
    "trends": ["temporal", "temporal_sensitivity", "dairy_meat"],
    "risk": ["risk"],
    "ecology": ["ecology", "clinical_exploratory"],
    "drivers": ["models", "shap_analysis", "responses", "associations"],
    "sensitivity": ["governance", "benchmark"],
    "figures": ["figure1", "figures", "supplementary_figures"],
    "figure1": ["figure1"],
}
ALL = [
    "figure1",
    "occurrence",
    "spatial_temporal_summaries",
    "temporal",
    "temporal_sensitivity",
    "dairy_meat",
    "risk",
    "ecology",
    "clinical_exploratory",
    "governance",
    "models",
    "shap_analysis",
    "responses",
    "associations",
    "benchmark",
    "descriptive",
    "figures",
    "supplementary_figures",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["all"] + list(GROUPS),
        default="all",
        help="Default: all main and supplementary analyses",
    )
    args = parser.parse_args()
    stages = ALL if args.step == "all" else GROUPS[args.step]
    if args.step != "figure1" and not INPUTS.is_dir():
        parser.error(
            "Place the companion frozen analysis inputs in data/ before running this step."
        )
    if args.step == "figures":
        source = Path(__file__).resolve().parent / "source_data"
        for path in source.iterdir():
            if path.name != "figure_data_index.csv" and path.suffix in {".csv", ".npz"}:
                target = INPUTS.parent / "outputs" / path.name
                if not target.exists():
                    shutil.copy2(path, target)
    start = time.time()
    with threadpool_limits(2):
        for name in stages:
            print("Running", name, flush=True)
            m = importlib.import_module("analysis." + name)
            m.run()
            if name == "ecology":
                m.curves()
    print(
        "Finished. Results are in outputs/. Elapsed seconds:",
        round(time.time() - start),
        flush=True,
    )


if __name__ == "__main__":
    main()
