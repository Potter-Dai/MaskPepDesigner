"""Dependency-light regression checks; these do not exercise GPU/model numerics.

Run with: python -m unittest discover -s tests -v
Function extraction lets NumPy/stdlib checks run without importing Uni-Core.
"""
import ast
import copy
import importlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def extracted(path, names, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    ns = {} if namespace is None else namespace
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class MigrationTests(unittest.TestCase):
    def sample_functions(self):
        return extracted('MPDesigner/inference/utils.py', ['build_sample_from_residx', 'collate_numpy_dict'], {
            'np': np, 'deepcopy': copy.deepcopy, 'logging': logging,
            'PDB_CHAIN_IDS': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'PSEUDO_RESIDUE': {'aatype': np.array(20)},
        })

    def test_repeated_reference_residue_has_independent_masks(self):
        ref = {('A', 1): {'aatype': np.array(0)}}
        f = self.sample_functions()['build_sample_from_residx']
        sample = f([[(None, 'A', 1, False, 1)], [('%', 'A', 1, True, 0)]], ref)
        np.testing.assert_array_equal(sample['chain_id'], [1, 2])
        np.testing.assert_array_equal(sample['seq_gen_mask'], [0, 1])
        np.testing.assert_array_equal(sample['ligmask'], [1, 0])
        self.assertEqual(set(ref[('A', 1)]), {'aatype'})

    def test_no_contig_tuple_unpacking(self):
        f = self.sample_functions()['build_sample_from_residx']
        result = f(None, {('A', 1): {'aatype': np.array(0)}})
        np.testing.assert_array_equal(result['seq_gen_mask'], [1])
        # This only checks sample construction; default spatial crop still needs hotspots.

    def test_denovo_peptide_mask(self):
        result = self.sample_functions()['build_sample_from_residx']([[(None, None, 0, True, 0)]])
        np.testing.assert_array_equal(result['ligmask'], [0])
        np.testing.assert_array_equal(result['frame_gen_mask'], [1])

    def test_job_rejects_zero_steps_and_replicas(self):
        ns = extracted('MPDesigner/inference/job.py', ['Job'], {'np': np, 're': re, 'hash': lambda *args: 1})
        for kwargs in ({'num_steps': 0}, {'num_steps': -1}, {'num_replica': 0}, {'num_replica': -1},
                       {'num_steps': 1.2}, {'num_replica': 1.2}, {'num_replica': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ns['Job']('example', contig='@12', **kwargs)

    def test_missing_configuration_is_actionable(self):
        ns = extracted('MPDesigner/model.py', ['mpdesigner_architecture'], {
            'register_model_architecture': lambda *args: lambda f: f,
        })
        for args in (types.SimpleNamespace(), types.SimpleNamespace(model_name=None)):
            with self.assertRaisesRegex(ValueError, '--model-name'):
                ns['mpdesigner_architecture'](args)
        args = types.SimpleNamespace(model_name='denovo')
        ns['mpdesigner_architecture'](args)
        self.assertEqual(args.model_name, 'denovo')

    def test_metric_reduction_is_repeatable(self):
        calls = []
        ns = extracted('MPDesigner/loss.py', ['MPDesignerLoss'], {
            'register_loss': lambda *args: lambda cls: cls,
            'UnicoreLoss': object,
            'metrics': types.SimpleNamespace(log_scalar=lambda *args, **kwargs: calls.append(args)),
        })
        logs = [{'sample_size': 1, 'sample_weight': 2, 'loss': 3, 'accuracy': .5}]
        original = copy.deepcopy(logs)
        ns['MPDesignerLoss'].reduce_metrics(logs)
        first = calls[:]
        calls.clear()
        ns['MPDesignerLoss'].reduce_metrics(logs)
        self.assertEqual(logs, original)
        self.assertEqual(calls, first)

    def test_dummy_collation(self):
        tree = ast.parse((ROOT/'MPDesigner/dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'collater')
        ns = {'DUMMY_BATCH': {'dummy': np.array([1])},
              'collate_dict': lambda samples, dim: samples,
              'data_ops': types.SimpleNamespace(get_pad_size=lambda n, p: n,
                  pad_to_size=lambda *args: lambda sample: sample)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'collater', 'exec'), ns)
        cfg = types.SimpleNamespace(data=types.SimpleNamespace(crop=types.SimpleNamespace(pad_size=4), batch_features={}))
        obj = types.SimpleNamespace(config=cfg)
        valid = {'aatype': np.array([1, 2])}
        self.assertEqual(ns['collater'](obj, [{'dummy': 1}, valid]), [valid])
        self.assertIn('dummy', ns['collater'](obj, [{'dummy': 1}])[0])
        self.assertEqual(ns['collater'](obj, []), [])

    def test_legacy_imports_resolve_once(self):
        # Stub only training registration dependencies; import actual package/contig modules.
        code = '''
import sys, types, importlib
for name in ('task', 'model', 'loss'):
    sys.modules['MPDesigner.' + name] = types.ModuleType('MPDesigner.' + name)
first = importlib.import_module(sys.argv[1] + '.inference.contig')
second = importlib.import_module(sys.argv[2] + '.inference.contig')
assert first is second
assert first.ContigParser is second.ContigParser
segments = first.ContigParser().parse('A1-10{3,5}:@12')
assert segments[1][0].is_ligand
'''
        for order in [('diffold', 'MPDesigner'), ('MPDesigner', 'diffold')]:
            subprocess.run([sys.executable, '-B', '-c', code, *order], cwd=ROOT, check=True)

    def test_training_registrations_accept_both_names(self):
        # Exercise actual decorated definitions with duplicate-name checks, without Uni-Core.
        for filename, base, current, legacy, decorator in (
            ('model.py', 'BaseUnicoreModel', 'MPDesignerModel', 'DiffoldModel', 'register_model'),
            ('task.py', 'UnicoreTask', 'MPDesignerTask', 'DiffoldTask', 'register_task'),
            ('loss.py', 'UnicoreLoss', 'MPDesignerLoss', 'DiffoldLoss', 'register_loss'),
        ):
            registry, class_names, architectures = {}, set(), {}
            def register(name):
                def apply(cls):
                    self.assertNotIn(name, registry)
                    self.assertNotIn(cls.__name__, class_names)
                    registry[name] = cls
                    class_names.add(cls.__name__)
                    return cls
                return apply
            def register_arch(model, arch):
                self.assertIn(model, registry)
                self.assertNotIn(arch, architectures)
                architectures[arch] = model
                return lambda f: f
            names = [current, legacy]
            if filename == 'model.py':
                names += ['mpdesigner_architecture', 'arch_v4']
            extracted('MPDesigner/' + filename, names, {
                base: object, decorator: register, 'register_model_architecture': register_arch,
            })
            self.assertTrue(issubclass(registry['diffold'], registry['MPDesigner']))
            if filename == 'model.py':
                self.assertEqual(architectures, {'MPDesigner_v6': 'MPDesigner', 'diffold_v6': 'diffold'})

    def test_launcher_propagates_failure_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            launcher = bin_dir/'python'
            launcher.write_text('#!/bin/bash\nwhile [ "$#" -gt 0 ]; do\n  if [ "$1" = --tmp-save-dir ]; then printf "%s" "$2" > "$CAPTURE"; fi\n  shift\ndone\nexit 7\n')
            launcher.chmod(0o755)
            unicore = bin_dir/'unicore-train'
            unicore.write_text('#!/bin/bash\nexit 0\n')
            unicore.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'], n_gpu='1', CAPTURE=str(bin_dir/'capture'))
            result = subprocess.run(['bash', 'train_diffold.sh', 'uf', 'distill', 'esm', str(bin_dir/'out'), 'denovo'], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertTrue((bin_dir/'out').is_dir())
            self.assertFalse(Path((bin_dir/'capture').read_text()).exists())

    def test_launchers_validate_argument_count(self):
        for script in ROOT.glob('*MPDesigner*.sh'):
            if script.name.startswith('.'):
                continue
            result = subprocess.run(['bash', str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, script.name)
            self.assertIn('Usage:', result.stderr)

    def test_setup_discovers_inference_losses_and_resource(self):
        from setuptools import find_packages
        packages = find_packages(where=str(ROOT))
        for name in ('MPDesigner.inference', 'MPDesigner.losses', 'diffold.inference', 'diffold.losses'):
            self.assertIn(name, packages)
        self.assertTrue((ROOT/'MPDesigner/data/plddt_spline.pkl.gz').is_file())

    def test_ssbuilder_zero_based_preserve_and_non_mutating_build(self):
        ns = extracted('SSbuilder/SSbuilder.py', [
            'Structure', 'SVDSuperimposer', 'SS_geo', 'superimpose',
            'dict2matrix', 'residues2atoms', 'cys_terminal', 'compare_ss',
            'cal_bfs_ss', 'build_ssbond'
        ], {
            'os': os, 'copy': copy, 'List': list, 'np': np,
            'dot': np.dot, 'transpose': np.transpose, 'sqrt': np.sqrt,
            'svd': np.linalg.svd, 'det': np.linalg.det,
            'aa': {'GLY': 'G', 'ALA': 'A', 'CYS': 'C'},
        })
        # Verify SS_geo maps 0-based index_preserve to the last residue (1-based resid=4)
        with tempfile.TemporaryDirectory() as tmp:
            pdb_path = Path(tmp) / 'pep.pdb'
            lines = []
            serial = 1
            for resid in range(1, 5):
                for aname, dx, dy, dz in (('N', 0.0, 0.0, 0.0), ('CA', 1.4, 0.0, 0.0),
                                          ('C', 2.4, 0.0, 0.0), ('O', 3.0, 0.0, 0.0),
                                          ('CB', 1.4, 1.0, 0.0)):
                    lines.append(
                        f"ATOM  {serial:5d} {aname:<4s} ALA B{resid:4d}    "
                        f"{resid*1.2+dx:8.3f}{dy:8.3f}{dz:8.3f}  1.00  1.00           C\n"
                    )
                    serial += 1
            pdb_path.write_text(''.join(lines))
            s = ns['Structure'](str(pdb_path))
            # Preserve all except resid 1 (index 0); resid 4 (index 3) is in index_preserve
            geo = ns['SS_geo'](s, index_preserve=[1, 2, 3]).geo
            self.assertEqual(geo, {})
            # Allow index 0 and index 3 (resid 1 and resid 4)
            geo2 = ns['SS_geo'](s, index_preserve=[1, 2]).geo
            self.assertIn('B_1_A@B_4_A', geo2)

    def test_find_matching_original_pdb(self):
        import glob
        ns = extracted('run_rescon_com.py', ['find_matching_original_pdb'], {'os': os, 'glob': glob})
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / '8qza_protein_A_peptide_B_107_132_loop.pdb'
            target.write_text('END\n')
            found = ns['find_matching_original_pdb'](
                'out_8qza_protein_A_peptide_B_107_132_loop_rep000_seed12345.pdb', tmp
            )
            self.assertEqual(found, str(target))
            # Also test task_id style filename matching unique 4-char PDB prefix
            found2 = ns['find_matching_original_pdb'](
                'out_8qza_denovo_s50_rep000_seed12345.pdb', tmp
            )
            self.assertEqual(found2, str(target))


if __name__ == '__main__':
    unittest.main()
