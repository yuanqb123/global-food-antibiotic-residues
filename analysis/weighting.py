"""Country, source, sampling-frame and underlying-record evidence weights."""

import numpy as np
import pandas as pd


def block_weights(g):
    # Equal total weight to each country, then DOI/frame block. This curbs
    # multi-analyte and publication-size dominance; n is retained within block.
    country = g.country_iso3.fillna("unknown").astype(str)
    block = (
        country
        + "|"
        + g.source_id.astype(str)
        + "|"
        + g.sampling_frame_id.fillna("unknown").astype(str)
    )
    blocks_per_country = block.groupby(country).transform("nunique").to_numpy(float)
    n_per_block = g.sample_size_n.groupby(block).transform("sum").to_numpy(float)
    w = 1000.0 / (blocks_per_country * n_per_block)
    return w


def weights(g, occurrence=False):
    if occurrence:
        return block_weights(g) * g.sample_size_n.to_numpy()
    co = g.country_iso3.fillna("unknown").astype(str)
    block = (
        co + "|" + g.source_id + "|" + g.sampling_frame_id.fillna("unknown").astype(str)
    )
    nblock = block.groupby(co).transform("nunique")
    nrecords = g.canonical_row.groupby(block).transform("nunique")
    nstat = g.canonical_row.groupby(g.canonical_row).transform("size")
    return (1 / (nblock * nrecords * nstat)).to_numpy(float)
