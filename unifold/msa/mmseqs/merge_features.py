
# author: Li, Ziyao <lizy01@dp.tech>
import gzip
import glob
import numpy as np
import pickle
import lmdb
import tqdm
import multiprocessing as mp
import os
from typing import *
from pathlib import Path
from argparse import ArgumentParser

import logging

from unifold.msa import parsers
from unifold.data import residue_constants as rc

from unifold.msa.pipeline import make_msa_features, make_sequence_features
from unifold.data.utils import compress_features
from unifold.data.data_ops import NumpyDict

logger = logging.getLogger(__name__)

def parse_lookup(path: str) -> Mapping[str, str]:
    with open(path) as f:
        lines = [l.rstrip().split('\t') for l in f]
    return {l[0]: l[1].replace('/', '') for l in lines}


def merge_features(inputs):
    a3m_path, templ_res_path, lmdb_path, out_path = inputs

    if not out_path:
        logger.warning(f"{a3m_path}")
    
    env = lmdb.open(lmdb_path, readonly=True, map_size=10_000_000)
    txn = env.begin()

    def _tax_map(x):
        uniprot_id = x.split('\t')[0]
        ret = txn.get(uniprot_id.encode())
        ret = ret.decode() if ret is not None else ""
        return ret

    msa = parsers.parse_a3m(open(a3m_path).read())
    msa_feats = make_msa_features([msa], _tax_map)
    txn.commit()

    seq, desc = msa.sequences[0], msa.descriptions[0]
    seq_feats = make_sequence_features(seq, desc, len(seq))

    if templ_res_path is not None and os.path.isfile(templ_res_path):
        templ_feats = pickle.load(gzip.open(templ_res_path))
        if (templ_feats["template_aatype"].shape == (0,)):
            templ_feats = {}

    else:
        print(templ_res_path, "does not exist. no template keys are returned.")
        templ_feats = {}
    
    all_feats = {**seq_feats, **msa_feats, **templ_feats}

    # check consistency
    try:
        assert all_feats["aatype"].shape[0] == all_feats["msa"].shape[1]
        if templ_feats:
            assert all_feats["aatype"].shape[0] == all_feats["template_aatype"].shape[1]
    except Exception as ex:
        print(a3m_path, "inconsistent", str(ex), (all_feats["aatype"].shape, all_feats["msa"].shape, all_feats["template_aatype"].shape if templ_feats else None))

    all_feats = compress_features(all_feats)

    with gzip.open(out_path, "wb") as f:
        pickle.dump(all_feats, f)
    
    return 0
    

def merge_feature_pipeline(
    a3m_dir: str,
    templ_res_dir: str,
    dbbase: str,
    out_dir: str,
    num_workers: int,
):
    os.makedirs(out_dir, exist_ok=True)
    lookup_path = os.path.join(a3m_dir, "query.lookup")
    lmdb_path = os.path.join(dbbase, "uniprot_tax.lmdb")

    a3m_paths = glob.glob(f"{a3m_dir}/*.a3m")
    num_tasks = len(a3m_paths)
    print(f"get {num_tasks} tasks: {a3m_paths[:3]} ...")
    a3m_paths = glob.glob(f"{a3m_dir}/*.a3m")
    lookup = parse_lookup(lookup_path)
    qid_nums = [Path(p).stem for p in a3m_paths]
    qids = [lookup.get(qn, None) for qn in qid_nums]
    templ_res_paths = [f"{templ_res_dir}/{q}.template.pkl.gz" if q else None for q in qids]
    lmdb_paths = [lmdb_path] * num_tasks
    out_paths = [f"{out_dir}/{q}.feature.pkl.gz" if q else None for q in qids]

    with mp.Pool(num_workers) as pool:
        [_ for _ in tqdm.tqdm(pool.imap(merge_features, zip(a3m_paths, templ_res_paths, lmdb_paths, out_paths)), total=num_tasks)]


if __name__ == "__main__":
    parser = ArgumentParser("make_features")
    parser.add_argument("mmseqs_res_dir", type=str, help=r"directory with `\d+.a3m` files.")
    parser.add_argument("templ_res_dir", type=str, help="directory with template <qid>.pkl.gz files.")
    parser.add_argument("dbbase", type=str, help="tsv for numId-2-qureyId map.")
    parser.add_argument("out_dir", type=str, help="directory with output files.")
    parser.add_argument("--num_workers", type=int, default=64)
    args = parser.parse_args()
    merge_feature_pipeline(
        args.mmseqs_res_dir,
        args.templ_res_dir,
        args.dbbase,
        args.out_dir,
        args.num_workers,
    )
