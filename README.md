# Mariner-Monitor — Marine Vessel Anomaly Detection

**Sector:** maritime operations  
**Task:** unsupervised anomaly scoring for synthetic vessel telemetry

Compares a robust per-feature deviation score with Isolation Forest. Training uses only normal rows from the training vessels; labels are reserved for held-out scoring.

## Data provenance and split

`mariner/data.py` generates synthetic AIS-like telemetry with injected speed, acceleration, course-change, navigation-error, and message-gap anomalies. It does not use real AIS or vessel records. The holdout is grouped by vessel so a vessel cannot appear in both train and test.

## Models and evaluation

Each model's alert threshold is the 95th percentile of its anomaly scores on normal training rows. The report includes ROC-AUC, average precision, precision, recall, confusion counts, false positives, and false negatives. That threshold targets a nominal 5% training false-alarm rate; test performance can differ.

## Limitations

The simulator does not include geographic routes, vessel types, maritime context, sensor outages, or real anomaly labels. It is not a maritime safety or enforcement tool. No operational vessel should be flagged from this prototype.

## Run

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m mariner.train --seed 42 --samples 3000 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` are shortcuts. Python 3.9 or newer is required.

## License status

No code license has been selected; default copyright rules apply. Choose a license before public distribution. No external data is included.
