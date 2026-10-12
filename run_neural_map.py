from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the first Project XYZ neural-map experiment pipeline.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model DIRECTORY")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--limit-per-category", type=int, default=5)
    parser.add_argument("--top-candidates", type=int, default=12)
    parser.add_argument("--categories", nargs="*", default=["general", "coding", "math", "reasoning", "knowledge"])
    args = parser.parse_args()

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    profile = str(artifacts / "profile.json")

    run([
        sys.executable,
        "profiler.py",
        "--model", args.model,
        "--prompts", args.prompts,
        "--output", profile,
        "--groups", str(args.groups),
        "--limit-per-category", str(args.limit_per_category),
    ])

    ablation_files = []
    for category in args.categories:
        output = str(artifacts / f"ablation_{category}.json")
        run([
            sys.executable,
            "ablation.py",
            "--model", args.model,
            "--profile", profile,
            "--prompts", args.prompts,
            "--target", category,
            "--output", output,
            "--top-candidates", str(args.top_candidates),
            "--limit-per-category", str(args.limit_per_category),
        ])
        ablation_files.append(output)

    run([
        sys.executable,
        "build_map.py",
        "--inputs",
        *ablation_files,
        "--output", str(artifacts / "neural_map.json"),
    ])

    print("\nDone. First experimental map: artifacts/neural_map.json")
    print("Do not treat activation or NLL sensitivity as proof that knowledge lives only in those blocks.")


if __name__ == "__main__":
    main()
