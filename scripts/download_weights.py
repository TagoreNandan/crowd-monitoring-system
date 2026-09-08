#!/usr/bin/env python3
"""Download pretrained crowd-counting weights.

Usage:
    python scripts/download_weights.py                       # default: DM-Count / QNRF
    python scripts/download_weights.py --model DM-Count --weights SHA
    python scripts/download_weights.py --all                 # everything in the registry
"""

from __future__ import annotations

import argparse
import sys

from crowd_monitoring.models.weights import AVAILABLE_WEIGHTS, ensure_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="DM-Count", choices=list(AVAILABLE_WEIGHTS))
    parser.add_argument("--weights", default="QNRF")
    parser.add_argument("--all", action="store_true", help="Download every available (model, weights) pair.")
    args = parser.parse_args()

    if args.all:
        pairs = [(m, w) for m, ws in AVAILABLE_WEIGHTS.items() for w in ws]
    else:
        if args.weights not in AVAILABLE_WEIGHTS[args.model]:
            print(
                f"'{args.weights}' not available for {args.model}. "
                f"Choices: {list(AVAILABLE_WEIGHTS[args.model])}",
                file=sys.stderr,
            )
            sys.exit(1)
        pairs = [(args.model, args.weights)]

    for model_name, model_weights in pairs:
        print(f"== {model_name} / {model_weights} ==")
        path = ensure_weights(model_name, model_weights)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
