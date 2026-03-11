"""
Drive build_curvature_single.py via subprocess to avoid igl memory accumulation.
Each graph is processed in an isolated Python process.

Usage:
    python scripts/build_curvature_batch.py
"""
import json
import subprocess
import sys
from pathlib import Path

graph_dir = Path("data/graphs")
reg_dir = Path("data/registered")
out_dir = Path("data/graphs_curvature")
out_dir.mkdir(exist_ok=True)

with open(graph_dir / "manifest.json") as f:
    manifest = json.load(f)

print(f"Processing {len(manifest)} graphs via subprocesses...")
n_ok = n_fail = n_skip = 0
new_manifest = []

for i, entry in enumerate(manifest):
    out_path = out_dir / entry["file"]
    if out_path.exists():
        # Skip only if the file was built with the updated script (has principal_dir1)
        try:
            d = torch.load(out_path, weights_only=False)
            if hasattr(d, "principal_dir1"):
                new_entry = dict(entry)
                new_entry["in_node_dim"] = 5
                new_manifest.append(new_entry)
                n_ok += 1
                continue
        except Exception:
            pass  # fall through to reprocess

    mesh_path = reg_dir / entry["subject_id"] / f"{entry['frame']}_{entry['struct']}.ply"
    if not mesh_path.exists():
        n_skip += 1
        continue

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_curvature_single.py",
            json.dumps(entry),
            str(graph_dir),
            str(reg_dir),
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 0:
        new_entry = dict(entry)
        new_entry["in_node_dim"] = 5
        new_manifest.append(new_entry)
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  [{n_ok}/{len(manifest)}] done", flush=True)
    else:
        print(f"  FAIL [{i}]: {entry['file']} | rc={result.returncode} | {result.stderr[:200]}", flush=True)
        n_fail += 1

with open(out_dir / "manifest.json", "w") as f:
    json.dump(new_manifest, f, indent=2)

print(f"\nDone: {n_ok} OK, {n_skip} skipped, {n_fail} failed")
print(f"Graphs saved to {out_dir}")
