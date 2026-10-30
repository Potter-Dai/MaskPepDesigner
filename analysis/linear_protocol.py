"""Strict per-design adapter around the existing DiffPepBuilder linear stages.

The historical run_postprocess.py and Postprocess defaults remain unchanged.
This adapter uses the same Amber relax parameters and original Rosetta XML,
without Uni-Fold as an extra runtime dependency or filename guesses.
"""
import math
import json
from pathlib import Path

from analysis.structure_io import require_complex


def run(input_pdb, output_dir, seed):
    input_pdb, output_dir = Path(input_pdb), Path(output_dir)
    peptide_length, receptor_length = require_complex(input_pdb)
    progress = {"status": "failed", "stage": "amber", "stages": {}}
    status_path = output_dir / "linear_status.json"
    try:
        from analysis.linear_amber import relax
        relaxed_text, amber_details = relax(input_pdb.read_text())
        amber_pdb = output_dir / "linear_amber_relaxed.pdb"
        amber_pdb.write_text(relaxed_text)
        require_complex(amber_pdb)
        progress["stages"]["amber"] = {"status": "ok", "output_pdb": str(amber_pdb),
                                        **amber_details}

        progress["stage"] = "rosetta_interface"
        import pyrosetta
        from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser
        xml = Path(__file__).with_name("interface_analyze.xml")
        pyrosetta.init("-mute all -ignore_zero_occupancy false -constant_seed -jran %d "
                       "-in:file:native %s" % (seed, str(input_pdb.resolve())))
        pose = pyrosetta.pose_from_pdb(str(amber_pdb))
        if pose.total_residue() != peptide_length + receptor_length or pose.num_chains() != 2:
            raise ValueError("unexpected linear complex topology")
        original_sequence = pose.sequence()
        RosettaScriptsParser().generate_mover(str(xml)).apply(pose)
        if pose.sequence() != original_sequence:
            raise ValueError("linear protocol changed peptide sequence")
        pyrosetta.get_fa_scorefxn()(pose)
        scores = dict(pose.scores)
        required = ("dG_separated", "dG_cross", "dSASA_int")
        if any(key not in scores or not math.isfinite(float(scores[key])) for key in required):
            raise ValueError("missing or non-finite linear interface metrics")
        if float(scores["dSASA_int"]) <= 0:
            raise ValueError("nonpositive linear interface area")
        final = output_dir / "linear_rosetta_relaxed.pdb"
        pose.dump_pdb(str(final))
        require_complex(final)
        from analysis.linear_geometry import audit
        geometry = audit(final)
        if not geometry["geometry_ok"]:
            raise ValueError("linear peptide bond or clash geometry failed")
        progress["stages"]["rosetta_interface"] = {
            "status": "ok", "xml": str(xml), "output_pdb": str(final)}
        result = {"final_pdb": str(final), "amber_pdb": str(amber_pdb),
                  "amber_details": amber_details, "stages": progress["stages"],
                  "peptide_length": peptide_length, "receptor_length": receptor_length,
                  "sequence_before": original_sequence[:peptide_length],
                  "sequence_after": pose.sequence()[:peptide_length],
                  "geometry": geometry,
                  "pyrosetta_version": pyrosetta.version(),
                  "protocol": "linear_diffpepbuilder_xml",
                  "dG_separated": float(scores["dG_separated"]),
                  "dG_cross": float(scores["dG_cross"]),
                  "dSASA_int": float(scores["dSASA_int"])}
        progress.update(status="ok", stage="complete", result=result)
        return result
    except Exception as exc:
        progress["reason"] = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        status_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
