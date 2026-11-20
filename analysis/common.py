from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def read(name):
    return pd.read_csv(INPUTS / name, low_memory=False)


def save(frame, name):
    target = OUTPUTS / (name if name.endswith(".csv") else name + ".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
