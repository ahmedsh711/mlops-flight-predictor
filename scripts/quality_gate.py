import argparse
import json
import sys
from pathlib import Path

from flight_predictor.config import (
    MAX_REGRESSION_ALLOWED,
    MODEL_INFO_PATH,
    PREV_BEST_PATH,
    R2_THRESHOLD,
)


def load_current_metrics(model_info_path: Path) -> dict:
    if not model_info_path.exists():
        print(f"ERROR: {model_info_path} not found. Did `flight-train` run?")
        sys.exit(1)
    with open(model_info_path) as f:
        return json.load(f)


def run_quality_gate(
    r2_floor: float = R2_THRESHOLD,
    max_regression: float = MAX_REGRESSION_ALLOWED,
    model_info_path: Path = MODEL_INFO_PATH,
    prev_best_path: Path = PREV_BEST_PATH,
) -> None:
    info = load_current_metrics(model_info_path)
    r2 = info["metrics"]["r2_inr_space"]
    rmse = info["metrics"]["rmse_inr"]
    mae = info["metrics"]["mae_inr"]

    print(f"\n{'=' * 50}")
    print("  Flight Price Model — Quality Gate")
    print(f"{'=' * 50}")
    print(f"  R²   = {r2:.4f}  (threshold: >= {r2_floor:.4f})")
    print(f"  RMSE = ₹{rmse:,.0f}")
    print(f"  MAE  = ₹{mae:,.0f}")
    print(f"{'=' * 50}\n")

    failures = []

    # Check 1: Absolute R² floor
    if r2 < r2_floor:
        failures.append(
            f"R² = {r2:.4f} is below the absolute floor of {r2_floor:.4f}. "
            f"Need {r2_floor - r2:.4f} more points."
        )

    # Check 2: Regression from previous best
    if prev_best_path.exists():
        try:
            prev_r2 = float(prev_best_path.read_text().strip())
            degradation = prev_r2 - r2
            if degradation > max_regression:
                failures.append(
                    f"Model regressed by {degradation:.4f} R² points "
                    f"(previous best: {prev_r2:.4f}, current: {r2:.4f}). "
                    f"Max allowed regression: {max_regression:.4f}."
                )
            print(f"  Previous best R²: {prev_r2:.4f}")
            print(
                f"  Regression check: {degradation:.4f} (allowed: {max_regression:.4f})"
            )
        except ValueError:
            print(
                f"  (Warning: Could not parse {prev_best_path} — skipping regression check)"
            )
    else:
        print("  (No previous best found — skipping regression check)")

    if failures:
        print("\n QUALITY GATE FAILED:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)  # Non-zero exit → CI pipeline fails

    # Update previous best if current is better
    should_update = False
    if not prev_best_path.exists():
        should_update = True
    else:
        try:
            prev_val = float(prev_best_path.read_text().strip() or "0")
            if r2 > prev_val:
                should_update = True
        except ValueError:
            should_update = True

    if should_update:
        prev_best_path.parent.mkdir(parents=True, exist_ok=True)
        prev_best_path.write_text(str(r2))
        print(f"  Updated previous best to {r2:.4f}")

    print("QUALITY GATE PASSED\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flight Price Prediction Quality Gate: Enforces R2 floor and regression tolerances."
    )
    parser.add_argument(
        "--r2-floor",
        type=float,
        default=R2_THRESHOLD,
        help=f"Minimum acceptable R2 score (default: {R2_THRESHOLD})",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=MAX_REGRESSION_ALLOWED,
        help=f"Max allowable R2 drop from previous best (default: {MAX_REGRESSION_ALLOWED})",
    )
    parser.add_argument(
        "--model-info-path",
        type=Path,
        default=MODEL_INFO_PATH,
        help=f"Path to model_info.json (default: {MODEL_INFO_PATH})",
    )
    parser.add_argument(
        "--prev-best-path",
        type=Path,
        default=PREV_BEST_PATH,
        help=f"Path to previous_best_r2.txt (default: {PREV_BEST_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_quality_gate(
        r2_floor=args.r2_floor,
        max_regression=args.max_regression,
        model_info_path=args.model_info_path,
        prev_best_path=args.prev_best_path,
    )
