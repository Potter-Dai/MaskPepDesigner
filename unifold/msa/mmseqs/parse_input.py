from typing import *
from pathlib import Path
import pandas   # TODO: remove this dependency.
from unifold.msa.parsers import parse_fasta
import random
from itertools import chain
import logging
logger = logging.getLogger(__name__)

import re


def clean_query_id(qid: str):
    # replace [^A-Za-z0-9\-] chars to _ and trunc to 10.
    qid = re.sub(r"[^A-Za-z0-9\-]", "_", qid)
    qid = re.sub(r"_+", "_", qid)
    qid = qid[:10]
    return qid

def parse_csv_queries(path: Path):
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pandas.read_csv(path, sep=sep)
    assert "id" in df.columns and "sequence" in df.columns
    queries = [
        (qid, seq, None)
        for qid, seq in df[["id", "sequence"]].itertuples(index=False)
    ]
    return queries

def parse_single_query(path: Path, use_file_stem_qid: bool = False):
    if path.suffix in (".fasta", ".faa", ".fa"):
        (seqs, headers) = parse_fasta(path.read_text())
        queries = [
            (hd, seq, None) for hd, seq in zip(headers, seqs)
        ]
    elif path.suffix == ".a3m":
        (seqs, headers) = parse_fasta(path.read_text())
        queries = [(headers[0], seqs[0], path.read_text())]
    else:   # no valid queries returned.
        return []
    
    if use_file_stem_qid:
        qid_fn = lambda x: clean_query_id(f"{path.stem}_{x}")
    else:
        qid_fn = lambda x: clean_query_id(x)
    
    return [(qid_fn(hd), s.upper(), a3m) for hd, s, a3m in queries]

def remove_duplicate_queries(queries):
    unique_qid_cnts = dict()
    unique_seq_a3m_pairs = set()
    unique_queries = []
    for qid, seq, a3m in queries:
        # reduce (seq, a3m) pairs.
        if (seq, a3m) in unique_seq_a3m_pairs:
            continue
        else:
            unique_seq_a3m_pairs.add((seq, a3m))
        # regenerate unique qids.
        if qid in unique_qid_cnts:
            qid = f"{qid}_{unique_qid_cnts[qid]:d}"
            unique_qid_cnts[qid] += 1
        else:
            unique_qid_cnts[qid] = 1
        unique_queries.append((qid, seq, a3m))
    return unique_queries

def get_queries(
    input_path: Union[str, Path],
    sort_queries_by: str = "len",
    use_file_stem_qid: bool = False,
) -> Tuple[List[Tuple[str, str, Optional[List[str]]]], bool]:
    """Reads a directory of fasta files, a single fasta file or a csv file and returns a tuple
    of job name, sequence and the optional a3m lines"""
    """This does not support multimer inputs anymore."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    if input_path.is_file():
        if input_path.suffix in (".csv", ".tsv"):
            queries = parse_csv_queries(input_path)
        elif input_path.suffix in (".fasta", ".faa", ".fa", ".a3m"):
            queries = parse_single_query(input_path, use_file_stem_qid)
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    elif input_path.is_dir():
        queries = [q for q in chain(
            *[parse_single_query(p) for p in input_path.iterdir()]
        )]

    # sort the queries
    if sort_queries_by == "len":
        queries.sort(key=lambda q: len(q[1]))
    elif sort_queries_by == "id":
        queries.sort(key=lambda q: q[0])
    elif sort_queries_by == "random":
        random.shuffle(queries)
    else:   # do nothing
        pass

    queries = remove_duplicate_queries(queries)
    
    return queries



def get_queries_old(
    input_path: Union[str, Path], sort_queries_by: str = "length"
) -> Tuple[List[Tuple[str, str, Optional[List[str]]]], bool]:
    """Reads a directory of fasta files, a single fasta file or a csv file and returns a tuple
    of job name, sequence and the optional a3m lines"""

    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    if input_path.is_file():
        if input_path.suffix == ".csv" or input_path.suffix == ".tsv":
            sep = "\t" if input_path.suffix == ".tsv" else ","
            df = pandas.read_csv(input_path, sep=sep)
            assert "id" in df.columns and "sequence" in df.columns
            queries = [
                (seq_id, sequence.upper().split(":"), None)
                for seq_id, sequence in df[["id", "sequence"]].itertuples(index=False)
            ]
            for i in range(len(queries)):
                if len(queries[i][1]) == 1: # monomer
                    queries[i] = (queries[i][0], queries[i][1][0], None)
        elif input_path.suffix == ".a3m":   # customized a3m
            raise NotImplementedError("pass the directory of a3m file(s) instead of a single one.")
        elif input_path.suffix in [".fasta", ".faa", ".fa"]:
            (sequences, headers) = parse_fasta(input_path.read_text())
            queries = []
            for sequence, header in zip(sequences, headers):
                sequence = sequence.upper()
                if sequence.count(":") == 0:
                    # Single sequence
                    queries.append((header, sequence, None))
                else:
                    # Complex mode
                    queries.append((header, sequence.upper().split(":"), None))
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    else:
        assert input_path.is_dir(), "Expected either an input file or a input directory"
        queries = []
        for file in sorted(input_path.iterdir()):
            if not file.is_file():
                continue
            if file.suffix.lower() not in [".a3m", ".fasta", ".faa"]:
                logger.warning(f"non-fasta/a3m file in input directory: {file}; ignored.")
                continue
            (seqs, header) = parse_fasta(file.read_text())
            if len(seqs) == 0:
                logger.error(f"{file} is empty")
                continue
            query_sequence = seqs[0]
            if len(seqs) > 1 and file.suffix in [".fasta", ".faa", ".fa"]:
                logger.warning(
                    f"More than one sequence in {file}, ignoring all but the first sequence"
                )

            if file.suffix.lower() == ".a3m":
                a3m_lines = [file.read_text()]
                queries.append((file.stem, query_sequence.upper(), a3m_lines))
            else:
                if query_sequence.count(":") == 0:
                    # Single sequence
                    queries.append((file.stem, query_sequence, None))
                else:
                    # Complex mode
                    queries.append((file.stem, query_sequence.upper().split(":"), None))

    # sort by seq. len
    if sort_queries_by == "length":
        queries.sort(key=lambda t: len(t[1]))
    elif sort_queries_by == "id":
        queries.sort(key=lambda t: t[0])
    elif sort_queries_by == "random":
        random.shuffle(queries)
    
    is_complex = False
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        if isinstance(query_sequence, list):
            is_complex = True
            break
        if a3m_lines is not None and a3m_lines[0].startswith("#"):
            a3m_line = a3m_lines[0].splitlines()[0]
            tab_sep_entries = a3m_line[1:].split("\t")
            if len(tab_sep_entries) == 2:
                query_seq_len = tab_sep_entries[0].split(",")
                query_seq_len = list(map(int, query_seq_len))
                query_seqs_cardinality = tab_sep_entries[1].split(",")
                query_seqs_cardinality = list(map(int, query_seqs_cardinality))
                is_single_protein = (
                    True
                    if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1
                    else False
                )
                if not is_single_protein:
                    is_complex = True
                    break
    
    return queries,  is_complex