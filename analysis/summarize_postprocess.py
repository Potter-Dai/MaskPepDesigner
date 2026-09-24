"""Summarize one-design status.json files without silently dropping failures."""
import argparse
import csv
import json
from pathlib import Path


FIELDS = ("design", "status", "stage", "reason", "topology", "protocol",
          "input_pdb", "final_pdb", "peptide_length", "sequence_before",
          "sequence_after", "dG_separated_reu", "dG_cross_reu", "dSASA_int_a2")


def summarize(input_dir, output):
    input_dir, output = Path(input_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    folders = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not folders:
        raise ValueError("no design directories found")
    rows = []
    for folder in folders:
        status = folder / "status.json"
        if status.is_file():
            try:
                record = json.loads(status.read_text())
            except (OSError, ValueError) as exc:
                record = {"status": "invalid", "reason": str(exc)}
        else:
            record = {"status": "missing", "reason": "status.json absent"}
        row = {key: "" for key in FIELDS}
        row["design"] = folder.name
        for key in ("status", "stage", "reason", "topology", "protocol",
                    "input_pdb", "final_pdb", "peptide_length", "sequence_before",
                    "sequence_after"):
            row[key] = record.get(key, "")
        if record.get("status") == "ok":
            if not record.get("final_pdb") or not Path(record["final_pdb"]).is_file():
                row["status"] = "invalid"
                row["reason"] = "final PDB missing"
            else:
                for source, target in (("dG_separated", "dG_separated_reu"),
                                       ("dG_cross", "dG_cross_reu"),
                                       ("dSASA_int", "dSASA_int_a2")):
                    row[target] = record.get(source, "")
                    if row[target] == "":
                        row["status"] = "invalid"
                        row["reason"] = "metric missing: " + source
        rows.append(row)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.input_dir, args.output)
    print("designs=%d ok=%d failed_or_invalid=%d" %
          (len(rows), sum(r["status"] == "ok" for r in rows),
           sum(r["status"] != "ok" for r in rows)))


if __name__ == "__main__":
    main()
