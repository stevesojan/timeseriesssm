"""Train and test TimeSeriesSSM on one TSLib dataset and horizon with the selected configuration of the paper.

    python examples/tslib_benchmark.py --data path/to/ETTh1.csv --horizon 96 --seed 2021

The preset is chosen from the file name (ETTh1.csv -> ETTh1, weather.csv -> Weather, electricity.csv -> ECL,
traffic.csv -> Traffic) unless --preset is given. The TSLib csv files are available from the Time-Series-Library
(https://github.com/thuml/Time-Series-Library). Errors are on the normalised scale, as in the paper. GPU training uses
non-deterministic CUDA kernels, so a run differs from the reported mean of three seeds by about 1%.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from timeseriesssm import Forecaster  # noqa: E402
from timeseriesssm.data import load_tslib  # noqa: E402

FILE_TO_PRESET = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2", "weather": "Weather",
                  "electricity": "ECL", "traffic": "Traffic"}
# TimeSeriesSSM in the paper (Table 2): test MSE / MAE, mean of seeds 2021-2023
REPORTED = {"ETTh1": {96: (0.3692, 0.3886), 192: (0.4217, 0.4216), 336: (0.4572, 0.4400), 720: (0.4674, 0.4648)},
            "ETTh2": {96: (0.2855, 0.3338), 192: (0.3608, 0.3851), 336: (0.4068, 0.4210), 720: (0.4082, 0.4338)},
            "ETTm1": {96: (0.3184, 0.3555), 192: (0.3573, 0.3801), 336: (0.3917, 0.4046), 720: (0.4537, 0.4446)},
            "ETTm2": {96: (0.1676, 0.2452), 192: (0.2317, 0.2885), 336: (0.2927, 0.3289), 720: (0.3898, 0.3877)},
            "Weather": {96: (0.1480, 0.1897), 192: (0.2001, 0.2393), 336: (0.2607, 0.2840), 720: (0.3390, 0.3369)},
            "ECL": {96: (0.1307, 0.2238), 192: (0.1483, 0.2395), 336: (0.1777, 0.2725), 720: (0.1923, 0.2873)},
            "Traffic": {96: (0.4328, 0.2669), 192: (0.4504, 0.2739), 336: (0.4675, 0.2813), 720: (0.4948, 0.2987)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="TSLib csv, e.g. ETTh1.csv")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--horizon", type=int, default=96, choices=(96, 192, 336, 720))
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None, help="optional path for the trained model")
    args = ap.parse_args()

    preset = args.preset or FILE_TO_PRESET[pathlib.Path(args.data).stem]
    fc = Forecaster.from_preset(preset, horizon=args.horizon, standardize=False, device=args.device,
                                train_overrides={"seed": args.seed})
    d = load_tslib(args.data, seq_len=fc.seq_len)  # official splits, scaler fitted on the training rows
    (tr, ttr), (va, tva), (te, tte) = d.split("train"), d.split("val"), d.split("test")
    t0 = time.time()
    fc.fit(tr, va, ttr, tva)
    res = fc.evaluate(te, tte)
    ref = REPORTED[preset][args.horizon]
    print(json.dumps({"preset": preset, "horizon": args.horizon, "seed": args.seed, "test_mse": res["mse"],
                      "test_mae": res["mae"], "test_windows": res["windows"], "paper_mse": ref[0],
                      "paper_mae": ref[1], "params": fc.n_params, "epochs": len(fc.history),
                      "seconds": round(time.time() - t0, 1)}, indent=1))
    if args.save:
        fc.save(args.save)


if __name__ == "__main__":
    main()
