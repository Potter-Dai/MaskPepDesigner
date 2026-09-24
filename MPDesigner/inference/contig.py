import re
import numpy as np
import itertools
from typing import *
from collections import namedtuple

Segment = namedtuple("Segment", ("symbol", "chain_id", "start", "end", "is_ligand", "hotspots"))


class SegmentError(ValueError):
    pass

class ContigParser(object):
    def __init__(
        self,
        chain_sep: str = ":",
        segment_sep: str = "/",
        chain_total_len_sep: str = ";",
        segment_pattern: str = r"^(@)?([\$\*\%])?([A-Za-z]+)?(\d+)(?:-(\d+))?(?:\{([\d,-]+)\})?$"
    ) -> None:
        self.chain_sep = chain_sep
        self.segment_sep = segment_sep
        self.chain_total_len_sep = chain_total_len_sep
        self.segment_pattern = segment_pattern

    # 新增函数：解析hotspot字符串
    def parse_hotspots(self, hotspot_str):
        if hotspot_str is None:
            return set()
        hotspots = set()
        for part in hotspot_str.split(','):
            if '-' in part:
                s, e = part.split('-')
                hotspots.update(range(int(s), int(e) + 1))
            else:
                hotspots.add(int(part))
        return hotspots

    def parse_segment(self, segment_str: str):
        match = re.match(self.segment_pattern, segment_str)
        if match:
            is_ligand, symbol, chain_id, start, end, hotspots_str = match.group(1, 2, 3, 4, 5, 6)
            if start is None:
                raise SegmentError(segment_str)
            start = int(start)
            end = int(end) if end is not None else start
            if end < start:
                raise SegmentError(segment_str)
            hotspots = self.parse_hotspots(hotspots_str)
            return Segment(symbol, chain_id, start, end, is_ligand=bool(is_ligand), hotspots=hotspots)
        else:
            raise SegmentError(segment_str)

    def parse(self, contig: str):
        # TODO: chain total len inputs
        if contig is None:
            return None
        chain_strs = contig.strip().split(self.chain_sep)
        chains = []
        for c in chain_strs:
            segment_strs = c.strip().split(self.segment_sep)
            segments = [
                self.parse_segment(s.strip())
                for s in segment_strs
            ]
            chains.append(segments)
        return chains

default_contig_parser = ContigParser()


def segment_to_residx(segment):
    symbol, chain_id, start, end, is_ligand, hotspots = segment
    if chain_id is None:
        l = np.random.randint(start, end+1)
        ret = [(symbol, None, i, is_ligand, 0) for i in range(l)]  # 生成链无hotspot
    else:
        ret = [(symbol, chain_id, i, is_ligand, 1 if i in hotspots else 0)
               for i in range(start, end + 1)]
    return ret


def contig_to_residx(contig):
    if contig is None:
        return None
    ret = []
    for chain in contig:
        ret.append(
            [*itertools.chain(*[segment_to_residx(s) for s in chain])]
        )
    return ret
