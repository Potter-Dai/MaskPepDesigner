# code from ColabFold, modified by Li, Ziyao <lizy01@dp.tech>

"""
Functionality for running mmseqs locally. Takes in a fasta file, outputs final.a3m

Note: Currently needs mmseqs compiled from source
"""

import json
import logging
import math
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import List, Union
import contextlib

from unifold.msa.mmseqs.parse_input import get_queries
# from mmseqs_pairing import link_paired_msas

logger = logging.getLogger(__name__)


def run_mmseqs(mmseqs: Path, params: List[Union[str, Path]], timings: dict = None, timing_key: dict = None) -> float:
    params_log = " ".join(str(i) for i in params)
    logger.info(f"Running {mmseqs} {params_log}")
    tic = time()
    ret = subprocess.check_call([mmseqs] + params)
    toc = time()
    if timings is not None:
        key = timing_key if timing_key is not None else params[0]
        timings[key] = toc - tic
    return ret


def mmseqs_search(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2202_db"),
    template_db: Path = Path(""),  # Unused by default
    metagenomic_db: Path = Path("colabfold_envdb_202108_db"),
    mmseqs: Path = Path("mmseqs"),
    use_env: bool = True,
    use_templates: bool = False,
    filter: bool = True,
    expand_eval: float = math.inf,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    num_iterations: int = 3,
    max_sequences_per_search: int = 10000,
    max_accept: int = 1000000,
    s: float = 8,
    db_load_mode: int = 2,
    threads: int = 32,
    timings: dict = None,
    remove_temp: bool = True
):
    """Run mmseqs with a local colabfold database set

    db1: uniprot db (UniRef30)
    db2: Template (unused by default)
    db3: metagenomic db (colabfold_envdb_202108 or bfd_mgy_colabfold, the former is preferred)
    """
    if filter:
        # 0.1 was not used in benchmarks due to POSIX shell bug in line above
        #  EXPAND_EVAL=0.1
        align_eval = 10
        qsc = 0.8
        max_accept = 100000

    used_dbs = [uniref_db]
    if use_templates:
        used_dbs.append(template_db)
    if use_env:
        used_dbs.append(metagenomic_db)
    
    for db in used_dbs:
        if not dbbase.joinpath(f"{db}.dbtype").is_file():
            raise FileNotFoundError(f"Database {db} does not exist")
        if (
            not dbbase.joinpath(f"{db}.idx").is_file()
            and not dbbase.joinpath(f"{db}.idx.index").is_file()
        ):
            logger.warning("Search does not use index")
            db_load_mode = 0
            dbSuffix1 = "_seq"
            dbSuffix2 = "_aln"
        else:
            dbSuffix1 = ".idx"
            dbSuffix2 = ".idx"

    # fmt: off
    # @formatter:off
    search_param = ["--num-iterations", str(num_iterations), "--db-load-mode", str(db_load_mode), "-a", "-s", str(s), "-e", "0.1", "--max-seqs", str(max_sequences_per_search),]
    filter_param = ["--filter-msa", str(filter), "--filter-min-enable", "1000", "--diff", str(diff), "--qid", "0.0,0.2,0.4,0.6,0.8,1.0", "--qsc", "0", "--max-seq-id", "0.95",]
    expand_param = ["--expansion-mode", "0", "-e", str(expand_eval), "--expand-filter-clusters", str(filter), "--max-seq-id", "0.95",]
    
    # uniref.
    tic_uniref = time()
    timings_uniref = {}
    run_mmseqs(mmseqs, ["search", base.joinpath("qdb"), dbbase.joinpath(uniref_db), base.joinpath("res"), base.joinpath("tmp"), "--threads", str(threads)] + search_param, timings_uniref)
    run_mmseqs(mmseqs, ["expandaln", base.joinpath("qdb"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"), base.joinpath("res"), dbbase.joinpath(f"{uniref_db}{dbSuffix2}"), base.joinpath("res_exp"), "--db-load-mode", str(db_load_mode), "--threads", str(threads)] + expand_param, timings_uniref)
    run_mmseqs(mmseqs, ["mvdb", base.joinpath("tmp/latest/profile_1"), base.joinpath("prof_res")])
    run_mmseqs(mmseqs, ["lndb", base.joinpath("qdb_h"), base.joinpath("prof_res_h")])
    run_mmseqs(mmseqs, ["align", base.joinpath("prof_res"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"), base.joinpath("res_exp"), base.joinpath("res_exp_realign"), "--db-load-mode", str(db_load_mode), "-e", str(align_eval), "--max-accept", str(max_accept), "--threads", str(threads), "--alt-ali", "10", "-a"], timings_uniref)
    run_mmseqs(mmseqs, ["filterresult", base.joinpath("qdb"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                        base.joinpath("res_exp_realign"), base.joinpath("res_exp_realign_filter"), "--db-load-mode",
                        str(db_load_mode), "--qid", "0", "--qsc", str(qsc), "--diff", "0", "--threads",
                        str(threads), "--max-seq-id", "1.0", "--filter-min-enable", "100"], timings_uniref)
    run_mmseqs(mmseqs, ["result2msa", base.joinpath("qdb"), dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
                        base.joinpath("res_exp_realign_filter"), base.joinpath("uniref.a3m"), "--msa-format-mode",
                        "6", "--db-load-mode", str(db_load_mode), "--threads", str(threads)] + filter_param, timings_uniref)
    if remove_temp:
        subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_exp_realign")])
        subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_exp")])
        subprocess.run([mmseqs] + ["rmdb", base.joinpath("res")])
        subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_exp_realign_filter")])
    toc_uniref = time() - tic_uniref
    logger.info(f"uniref done in {toc_uniref/60:.2f} min.")
    timings_uniref["total"] = toc_uniref

    if use_templates:   # templates.
        tic_templ = time()
        timings_templ = {}
        run_mmseqs(mmseqs, ["search", base.joinpath("prof_res"), dbbase.joinpath(template_db), base.joinpath("res_pdb"), base.joinpath("tmp"), "--db-load-mode", str(db_load_mode), "--threads", str(threads), "-s", "7.5", "-a", "-e", "0.1"], timings_templ)
        # # if one wants to output template candidate a3ms from mmseqs, enable this.
        # run_mmseqs(mmseqs, ["result2msa", base.joinpath("qdb"), dbbase.joinpath(f"{template_db}{dbSuffix1}"),
        #                 base.joinpath("res_pdb"), base.joinpath("template.a3m"), "--msa-format-mode",
        #                 "6", "--db-load-mode", str(db_load_mode), "--threads", str(threads)] + filter_param, timings_templ)
        run_mmseqs(mmseqs, ["convertalis", base.joinpath("prof_res"), dbbase.joinpath(f"{template_db}{dbSuffix1}"), base.joinpath("res_pdb"), base.joinpath(f"{template_db}.m8"), "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar", "--db-load-mode", str(db_load_mode), "--threads", str(threads)], timings_templ)
        if remove_temp:
            subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_pdb")])
        toc_templ = time() - tic_templ
        logger.info(f"template search done in {toc_templ/60:.2f} min.")
        timings_templ["total"] = toc_templ
    
    if use_env:     # envdb.
        tic_envdb = time()
        timings_envdb = {}
        run_mmseqs(mmseqs, ["search", base.joinpath("prof_res"), dbbase.joinpath(metagenomic_db), base.joinpath("res_env"), base.joinpath("tmp"), "--threads", str(threads)] + search_param, timings_envdb)
        run_mmseqs(mmseqs, ["expandaln", base.joinpath("prof_res"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"), base.joinpath("res_env"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix2}"), base.joinpath("res_env_exp"), "-e", str(expand_eval), "--expansion-mode", "0", "--db-load-mode", str(db_load_mode), "--threads", str(threads)], timings_envdb)
        run_mmseqs(mmseqs,
                   ["align", base.joinpath("tmp/latest/profile_1"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                    base.joinpath("res_env_exp"), base.joinpath("res_env_exp_realign"), "--db-load-mode",
                    str(db_load_mode), "-e", str(align_eval), "--max-accept", str(max_accept), "--threads",
                    str(threads), "--alt-ali", "10", "-a"], timings_envdb)
        run_mmseqs(mmseqs, ["filterresult", base.joinpath("qdb"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign"), base.joinpath("res_env_exp_realign_filter"),
                            "--db-load-mode", str(db_load_mode), "--qid", "0", "--qsc", str(qsc), "--diff", "0",
                            "--max-seq-id", "1.0", "--threads", str(threads), "--filter-min-enable", "100"], timings_envdb)
        run_mmseqs(mmseqs, ["result2msa", base.joinpath("qdb"), dbbase.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign_filter"),
                            base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m"), "--msa-format-mode", "6",
                            "--db-load-mode", str(db_load_mode), "--threads", str(threads)] + filter_param, timings_envdb)
        if remove_temp:
            subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_env_exp_realign_filter")])
            subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_env_exp_realign")])
            subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_env_exp")])
            subprocess.run([mmseqs] + ["rmdb", base.joinpath("res_env")])
        toc_envdb = time() - tic_envdb
        logger.info(f"envdb done in {toc_envdb/60:.2f} min.")
        timings_envdb["total"] = toc_envdb

    if use_env:
        run_mmseqs(mmseqs, ["mergedbs", base.joinpath("qdb"), base.joinpath("final.a3m"), base.joinpath("uniref.a3m"), base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        if remove_temp:
            subprocess.run([mmseqs] + ["rmdb", base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
    else:
        run_mmseqs(mmseqs, ["mvdb", base.joinpath("uniref.a3m"), base.joinpath("final.a3m")])
    
    run_mmseqs(mmseqs, ["unpackdb", base.joinpath("final.a3m"), base.joinpath("."), "--unpack-name-mode", "0", "--unpack-suffix", ".a3m"])
    # # if one wants to output template candidate a3ms from mmseqs, enable this.
    # if use_templates:
    #     run_mmseqs(mmseqs, ["unpackdb", base.joinpath("template.a3m"), base.joinpath("."), "--unpack-name-mode", "0", "--unpack-suffix", ".template.a3m"])

    if remove_temp:
        subprocess.run([mmseqs] + ["rmdb", base.joinpath("final.a3m")])
        subprocess.run([mmseqs] + ["rmdb", base.joinpath("uniref.a3m")])
        subprocess.run([mmseqs] + ["rmdb", base.joinpath("res")])
        # @formatter:on
        # fmt: on
        for file in base.glob("prof_res*"):
            file.unlink()
        shutil.rmtree(base.joinpath("tmp"))

    if timings is not None:
        timings.update({
            "uniref": timings_uniref,
            "template": timings_templ if use_templates else None,
            "envdb": timings_envdb if use_env else None,
        })

    return 


def main(args):
    timings = {} if args.timing else None

    tic = time()
    queries = get_queries(args.query)
    args.base.mkdir(exist_ok=True, parents=True)

    query_fasta = args.base.joinpath("query.fasta")
    with query_fasta.open("w") as f:
        for qid, seq, a3m in queries:
            f.write(f">{qid}{'+a3m' if a3m is not None else ''}\n{seq}\n")

    run_mmseqs(
        args.mmseqs,
        ["createdb", query_fasta, args.base.joinpath("qdb"), "--shuffle", "0"],
        timings
    )

    with args.base.joinpath("query.lookup").open("w") as f:
        for i, (qid, _, _) in enumerate(queries):
            f.write(f"{i}\t{qid}\n")
    
    mmseqs_search(
        mmseqs=args.mmseqs,
        dbbase=args.dbbase,
        base=args.base,
        uniref_db=args.db1,
        template_db=args.db2,
        metagenomic_db=args.db3,
        use_env=args.use_env,
        use_templates=args.use_templates,
        filter=args.filter,
        expand_eval=args.expand_eval,
        align_eval=args.align_eval,
        diff=args.diff,
        qsc=args.qsc,
        num_iterations=args.num_iterations,
        max_accept=args.max_accept,
        s=args.s,
        db_load_mode=args.db_load_mode,
        threads=args.threads,
        timings=timings,
        remove_temp=args.remove_temp
    )

    if args.remove_temp:
        # query_fasta.unlink()      # do not do this.
        run_mmseqs(args.mmseqs, ["rmdb", args.base.joinpath("qdb")])
        run_mmseqs(args.mmseqs, ["rmdb", args.base.joinpath("qdb_h")])

    toc = time() - tic
    logger.info(f"{len(queries)} query sequences done in {toc/60:.2f} mins.")

    if args.timing:
        timings["total"] = toc
        with args.base.joinpath("timings.json").open("w") as f:
            json.dump(timings, f, indent=2)

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "query",
        type=Path,
        help="Query sequences. A fasta file, or a directory of fasta files "
             "are supported. CSV (TSV) files with columns `id` and "
             "`sequence` are supported.",
    )
    parser.add_argument(
        "dbbase",
        type=Path,
        help="The path to the database and indices you downloaded and created with setup_databases.sh",
    )
    parser.add_argument(
        "base", type=Path, help="Directory for the results (and intermediate files)"
    )
    parser.add_argument(
        "-s",
        type=int,
        default=8,
        help="mmseqs sensitivity. Lowering this will result in a much faster search but possibly sparser msas",
    )
    # dbs are uniref, templates and environmental
    parser.add_argument(
        "--db1", type=Path, default=Path("uniref30_2202_db"), help="UniRef database"
    )
    parser.add_argument("--db2", type=Path, default=Path("pdb70_220313"), help="Templates database")
    parser.add_argument(
        "--db3",
        type=Path,
        default=Path("colabfold_envdb_202108_db"),
        help="Environmental database",
    )
    # poor man's boolean arguments
    parser.add_argument("--use-env", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-templates", type=int, default=1, choices=[0, 1])
    parser.add_argument("--filter", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--mmseqs",
        type=Path,
        default=Path("mmseqs"),
        help="Location of the mmseqs binary",
    )
    parser.add_argument("--expand-eval", type=float, default=math.inf)
    parser.add_argument("--align-eval", type=int, default=10)
    parser.add_argument("--diff", type=int, default=3000)
    parser.add_argument("--qsc", type=float, default=-20.0)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--max-accept", type=int, default=1000000)
    parser.add_argument("--db-load-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--timing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--remove-temp", type=int, default=1, choices=[0, 1])
    
    args = parser.parse_args()

    main(args)
