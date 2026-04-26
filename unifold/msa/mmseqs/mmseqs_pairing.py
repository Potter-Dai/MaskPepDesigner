raise NotImplementedError("do not use this script.")


import logging
import shutil
from pathlib import Path
from time import time

from search import run_mmseqs

logger = logging.getLogger(__name__)


def pad_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    _blank_seq = [
        ("-" * len(seq))
        for n, seq in enumerate(query_sequences)
        for _ in range(query_cardinality[n])
    ]
    a3m_lines_combined = []
    pos = 0
    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0:
                    continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append(
                        "".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1 :])
                    )
            pos += 1
    return "\n".join(a3m_lines_combined)


def pair_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
    for n, seq in enumerate(query_sequences):
        lines = a3m_lines[n].splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] = a3m_line_paired[i] + line
            else:
                a3m_line_paired[i] = a3m_line_paired[i] + line * query_cardinality[n]
    return "\n".join(a3m_line_paired)


def pair_msa(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    paired_msa: Optional[List[str]],
    unpaired_msa: Optional[List[str]],
) -> str:
    if paired_msa is None and unpaired_msa is not None:
        a3m_lines = pad_sequences(
            unpaired_msa, query_seqs_unique, query_seqs_cardinality
        )
    elif paired_msa is not None and unpaired_msa is not None:
        a3m_lines = (
            pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
            + "\n"
            + pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
        )
    elif paired_msa is not None and unpaired_msa is None:
        a3m_lines = pair_sequences(
            paired_msa, query_seqs_unique, query_seqs_cardinality
        )
    else:
        raise ValueError(f"Invalid pairing")
    return a3m_lines


def msa_to_str(
    unpaired_msa: List[str],
    paired_msa: List[str],
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
) -> str:
    msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
    msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
    # build msa with cardinality of 1, it makes it easier to parse and manipulate
    query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
    msa += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)
    return msa


def mmseqs_search_pair(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2202_db"),
    mmseqs: Path = Path("mmseqs"),
    s: float = 8,
    threads: int = 64,
    db_load_mode: int = 2,
):
    if not dbbase.joinpath(f"{uniref_db}.dbtype").is_file():
        raise FileNotFoundError(f"Database {uniref_db} does not exist")
    if (
        not dbbase.joinpath(f"{uniref_db}.idx").is_file()
        and not dbbase.joinpath(f"{uniref_db}.idx.index").is_file()
    ):
        logger.warning("Search does not use index")
        db_load_mode = 0
        dbSuffix1 = "_seq"
        dbSuffix2 = "_aln"
    else:       # this is activated in default.
        dbSuffix1 = ".idx"
        dbSuffix2 = ".idx"

    search_param = [
        "--num-iterations",
        "3",
        "--db-load-mode",
        str(db_load_mode),
        "-a",
        "-s",
        str(s),
        "-e",
        "0.1",
        "--max-seqs",
        "10000",
    ]
    expand_param = [
        "--expansion-mode",
        "0",
        "-e",
        "inf",
        "--expand-filter-clusters",
        "0",
        "--max-seq-id",
        "0.95",
    ]

    timings = {}

    timings["search"] = run_mmseqs(
        mmseqs,
        [
            "search",
            base.joinpath("qdb"),
            dbbase.joinpath(uniref_db),
            base.joinpath("res"),
            base.joinpath("tmp"),
            "--threads",
            str(threads),
        ]
        + search_param,
    )
    timings["expandaln"] = run_mmseqs(
        mmseqs,
        [
            "expandaln",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
            base.joinpath("res"),
            dbbase.joinpath(f"{uniref_db}{dbSuffix2}"),
            base.joinpath("res_exp"),
            "--db-load-mode",
            str(db_load_mode),
            "--threads",
            str(threads),
        ]
        + expand_param,
    )
    timings["align"] = run_mmseqs(
        mmseqs,
        [
            "align",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
            base.joinpath("res_exp"),
            base.joinpath("res_exp_realign"),
            "--db-load-mode",
            str(db_load_mode),
            "-e",
            "0.001",
            "--max-accept",
            "1000000",
            "--threads",
            str(threads),
            "-c",
            "0.5",
            "--cov-mode",
            "1",
        ],
    )
    timings["pairaln"] = run_mmseqs(
        mmseqs,
        [
            "pairaln",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{uniref_db}"),
            base.joinpath("res_exp_realign"),
            base.joinpath("res_exp_realign_pair"),
            "--db-load-mode",
            str(db_load_mode),
            "--threads",
            str(threads),
        ],
    )
    timings["align2"] = run_mmseqs(
        mmseqs,
        [
            "align",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
            base.joinpath("res_exp_realign_pair"),
            base.joinpath("res_exp_realign_pair_bt"),
            "--db-load-mode",
            str(db_load_mode),
            "-e",
            "inf",
            "--threads",
            str(threads),
        ],
    )
    timings["pairaln2"] = run_mmseqs(
        mmseqs,
        [
            "pairaln",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{uniref_db}"),
            base.joinpath("res_exp_realign_pair_bt"),
            base.joinpath("res_final"),
            "--db-load-mode",
            str(db_load_mode),
            "--threads",
            str(threads),
        ],
    )
    timings["result2msa"] = run_mmseqs(
        mmseqs,
        [
            "result2msa",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{uniref_db}{dbSuffix1}"),
            base.joinpath("res_final"),
            base.joinpath("pair.a3m"),
            "--db-load-mode",
            str(db_load_mode),
            "--msa-format-mode",
            "5",
            "--threads",
            str(threads),
        ],
    )
    timings["unpackdb"] = run_mmseqs(
        mmseqs,
        [
            "unpackdb",
            base.joinpath("pair.a3m"),
            base.joinpath("."),
            "--unpack-name-mode",
            "0",
            "--unpack-suffix",
            ".paired.a3m",
        ],
    )
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("qdb")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("qdb_h")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_pair")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_pair_bt")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_final")])
    run_mmseqs(mmseqs, ["rmdb", base.joinpath("pair.a3m")])
    shutil.rmtree(base.joinpath("tmp"))
    return timings

def link_paired_msas(queries_unique, base_path: Path):
    pt = time()
    id = 0
    for job_number, (
        raw_jobname,
        query_sequences,
        query_seqs_cardinality,
    ) in enumerate(queries_unique):
        unpaired_msa = []
        paired_msa = None
        if len(query_seqs_cardinality) > 1:
            paired_msa = []
        for seq in query_sequences:
            with base_path.joinpath(f"{id}.a3m").open("r") as f:
                unpaired_msa.append(f.read())
            base_path.joinpath(f"{id}.a3m").unlink()
            if len(query_seqs_cardinality) > 1:
                with base_path.joinpath(f"{id}.paired.a3m").open("r") as f:
                    paired_msa.append(f.read())
            base_path.joinpath(f"{id}.paired.a3m").unlink()
            id += 1
        msa = msa_to_str(
            unpaired_msa, paired_msa, query_sequences, query_seqs_cardinality
        )
        base_path.joinpath(f"{job_number}.a3m").write_text(msa)
    return time() - pt