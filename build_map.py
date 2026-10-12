from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Combine ablation results into a first neural circuit map.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Ablation JSON files")
    parser.add_argument("--output", default="artifacts/neural_map.json")
    parser.add_argument("--min-causal-specificity", type=float, default=0.0)
    parser.add_argument("--top-per-category", type=int, default=16)
    args = parser.parse_args()

    mapping = {
        "schema_version": 1,
        "description": "Experimental task-to-contiguous-MLP-group causal map. Not a proof of isolated knowledge storage.",
        "categories": {},
    }

    for path in args.inputs:
        data = load_json(path)
        category = data["target"]
        kept = []
        for row in data.get("results", []):
            if float(row.get("causal_specificity", 0.0)) < args.min_causal_specificity:
                continue
            kept.append(
                {
                    "layer": int(row["layer"]),
                    "group": int(row["group"]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "width": int(row["width"]),
                    "activation_specificity": float(row.get("specificity_score", 0.0)),
                    "causal_specificity": float(row.get("causal_specificity", 0.0)),
                    "target_delta_nll": float(row.get("target_delta_nll", 0.0)),
                    "control_delta_nll": float(row.get("control_delta_nll", 0.0)),
                }
            )
        kept.sort(key=lambda x: x["causal_specificity"], reverse=True)
        mapping["categories"][category] = kept[: args.top_per_category]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"Wrote {args.output}")
    for category, rows in mapping["categories"].items():
        print(f"  {category}: {len(rows)} mapped groups")


if __name__ == "__main__":
    main()
