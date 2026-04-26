from tqdm import tqdm
import logging
import os
from typing import *
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Pool
import pickle, gzip


logger = logging.getLogger(__name__)

from unifold.msa import templates, parsers
from unifold.msa.tools import hhsearch
from unifold.data import residue_constants
from collections import OrderedDict as odict

FFindex = Mapping[str, Tuple]
FFindexList = List[Tuple]
M8Hits = List[List[Any]]

def parse_template_m8(path: str) -> M8Hits:
    # "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar"
    templs = odict()
    for line in open(path):
        (qid, pid, fident, alnlen, mismatch, gapopen, qstart, qend, tstart, tend, evalue, bits, cigar) = line.rstrip().split()
        if qid not in templs:
            templs[qid] = []
        templs[qid].append(
            [pid, int(alnlen), int(qstart), int(qend), int(tstart), int(tend), float(evalue)]
        )
    templs = [(k, v) for k, v in templs.items()]
    return templs

def parse_lookup(path: str) -> Mapping[str, str]:
    with open(path) as f:
        lines = [l.rstrip().split('\t') for l in f]
    return {l[1]: l[0] for l in lines}

def filter_ffindex(ffindex: FFindex, targets: List[str]) -> FFindexList:
    targets.sort()  # sorted ffindex id
    return [(t, ffindex.get(t, (None, None))) for t in targets]

def parse_ffindex(path: str, skip_chains: List[str] = ["6WOV_C"]) -> FFindex:
    # 6WOV_C is a bad case in pdb70_from_mmcif_220313....
    ret = {}
    with open(path) as f:
        for l in f:
            name, offset, size = l.strip().split('\t')
            ret[name] = (offset, size)
    for c in skip_chains:
        if c in ret:
            ret.pop(c)
    return ret

def dump_ffindex(ffindex_list: FFindexList, path: str):
    # assert not os.path.isfile(path), f"{path} exists."
    with open(path, 'w') as f:
        for i, (o, s) in ffindex_list:
            if o is not None:
                f.write(f"{i}\t{o}\t{s}\n")
            else:
                logging.warning(f"template candidate {i} not in template database.")

def get_query_seq_from_a3m(a3m_path):
    with open(a3m_path) as f:
        desc = f.readline().rstrip().lstrip(">") # skip annotation
        query_seq = ""
        line = f.readline().rstrip()
        while not line.startswith(">"):
            query_seq += line
            line = f.readline().rstrip()
    query_seq = query_seq.replace("-", "")  # remove gaps
    return desc, query_seq

def make_pdb70_subset(templs: M8Hits, pdb70_dataset: str, tmp_dir: str) -> List[Tuple[str]]:
    
    ffindex = parse_ffindex(f"{pdb70_dataset}_a3m.ffindex")
    ffdata_path = f"{pdb70_dataset}_a3m.ffdata"

    templ_paths = []

    for qid, hits in templs:
        cur_dir = os.path.join(tmp_dir, f"template_{qid}")
        if os.path.isdir(cur_dir):
        # if os.path.isfile(f"{cur_dir}/READY"):  # template files already generated
            templ_paths.append((qid, cur_dir))
            continue
        os.makedirs(cur_dir, exist_ok=True)
        pids = [h[0] for h in hits]
        cur_ffindex = filter_ffindex(ffindex, pids)
        
        dump_ffindex(cur_ffindex, f"{cur_dir}/subset_a3m.ffindex")
        os.system(f"ln -s {ffdata_path} {cur_dir}/subset_a3m.ffdata")
        os.system(f"cp {cur_dir}/subset_a3m.ffindex {cur_dir}/subset_cs219.ffindex")
        os.system(f"touch {cur_dir}/subset_cs219.ffdata")
        os.system(f"touch {cur_dir}/READY")
        templ_paths.append((qid, cur_dir))
    
    return templ_paths


def worker(
    kwargs
):
    
    query_id: str = kwargs["query_id"]
    a3m_path: str = kwargs["a3m_path"]
    template_path: str = kwargs["template_path"]
    mmcif_dir: str = kwargs["mmcif_dir"]
    output_path: str = kwargs["output_path"]
    max_template_date: str = kwargs["max_template_date"]
    max_hit: str = kwargs["max_hit"]

    if output_path and os.path.isfile(output_path):
        logging.info(f"{output_path} already exists.")
        return None
    
    if not a3m_path:
        logging.warning(f"cannot find a3m file for query {query_id}")
        return None
    
    try:
        feat = get_template(a3m_path, template_path, mmcif_dir, max_template_date, max_hit)
    except Exception as ex:
        if output_path:
            with open(f"{output_path}.failed", "w") as f:
                f.write(f"{ex.__class__}\t{ex}\n")
        return None
    if output_path:
        with gzip.open(output_path, 'wb') as f:
                pickle.dump(feat, f)
    
    return feat


def template_pipeline(
    result_dir: str,
    dbbase: str,
    output_dir: Optional[str] = None,   # write out
    tmp_dir: Optional[str] = None,
    max_template_date: str = "2100-01-01",
    max_hit: int = 20,
    num_workers: int = 12,
):
    pdb70_dataset = os.path.join(dbbase, "pdb70")
    mmcif_dir = os.path.join(dbbase, "pdb_mmcif", "mmcif_files")
    assert not os.path.isdir(tmp_dir), f"{tmp_dir} occupied."
    if not tmp_dir:
        assert output_dir is not None
        tmp_dir = f"{output_dir}/tmp"
    templs = parse_template_m8(f"{result_dir}/pdb70_220313.m8")         # [(qid1, [[res1], [res2], ...]), (qid2, ...)]
    templ_paths = make_pdb70_subset(templs, pdb70_dataset, tmp_dir)
    lookup = parse_lookup(f"{result_dir}/query.lookup")

    def qid_to_a3m_path(qid):
        qid_num = lookup.get(qid, None)
        if qid_num:
            return f"{result_dir}/{qid_num}.a3m"
        else:
            return None
    
    def qid_to_output_path(qid):
        return f"{output_dir}/{qid}.template.pkl.gz"
    
    worker_inputs = [
        {
            "query_id": r[0],
            "a3m_path": qid_to_a3m_path(r[0]),
            "template_path": r[1],
            "mmcif_dir": mmcif_dir,
            "output_path": qid_to_output_path(r[0].replace("/", "-")),
            "max_template_date": max_template_date,
            "max_hit": max_hit,
        } for r in templ_paths
    ]

    with Pool(num_workers) as pool:
        rets = []
        for ret in tqdm(pool.imap(worker, worker_inputs), total=len(worker_inputs)):
            rets.append(ret)
    
    os.system(f"rm -rf {tmp_dir}")

    return rets


def get_template(
    a3m_path: str,
    template_path: str,
    mmcif_dir: str,
    max_template_date: str = "2100-01-01",
    max_hit: int = 20,
) -> Dict[str, Any]:
    description, query_sequence = get_query_seq_from_a3m(a3m_path)
    # print(query_sequence)

    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=mmcif_dir,
        max_template_date=max_template_date,
        max_hits=max_hit,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
        max_subsequence_ratio=1.0,  # do not skip identical subsequences.
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/subset"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(open(a3m_path).read())
    hhsearch_hits = parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)


def get_null_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, residue_constants.atom_type_num))
    templates_aatype = residue_constants.sequence_to_onehot(
        output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "mmseqs_res_dir",
        type=str,
        help="MMseqs2 result directory",
    )
    parser.add_argument(
        "dbbase",
        type=str,
        help="The path to the databases",
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory for the results"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="",
        help="Temporary directory that does not exist",
    )
    parser.add_argument(
        "--max-template-date",
        type=str,
        default="2100-01-01",
        help="Template cutoff date. Default: future"
    )
    parser.add_argument(
        "--max-hits", type=int, default=20, help="Max template hits")
    parser.add_argument(
        "--num-workers", type=int, default=64
    )
    args = parser.parse_args()

    template_pipeline(
        args.mmseqs_res_dir,
        args.dbbase,
        args.output_dir,
        args.tmp_dir,
        max_template_date=args.max_template_date,
        max_hit=args.max_hits,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()
