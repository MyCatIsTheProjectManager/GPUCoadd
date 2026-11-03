"""Conservative tile-memory estimates. No CUDA allocation is made by the planner.

The stack buffer is reserved for the busiest tile for the entire run. Scratch
estimates cover separate warp, coverage and integration phases, not their sum.
The estimate is a guard against paging, not a promise about allocator behaviour.
"""

GIB = 1 << 30
RESERVE = GIB
INTEGRATION_BYTES = {"sigma": 28, "winsorized": 32, "noise": 54}


def memory_estimate(tile, width, height, counts, crop_pixels=None, debayer="rcd", rejection="sigma", feather=False):
    area = int(min(tile, width) * min(tile, height))
    busiest = int(max(counts, default=0))
    buffer = busiest * (4 if feather else 3) * area * 4  # feathering: one weight per light and pixel more
    if not busiest:
        return {"buffer_bytes": 0, "scratch_bytes": 0, "required_bytes": RESERVE}
    # RCD uses batches capped at two million source pixels, except a single crop
    # may exceed that. Default padding/scratch bounds are calibrated against the
    # RTX 3060 trials; crop_pixels can supply a measured padded batch bound.
    pixels = crop_pixels if crop_pixels is not None else min(busiest * area * 4, max(4_000_000, area * 4))
    # Live RCD intermediates plus allocator slack for differently padded batches.
    # The original 256 B/pixel estimate understated measured scratch allocation.
    # The measured RCD scratch peak grows with larger tile area even though its
    # nominal batch pixel cap is fixed (padding/intermediate allocation shapes).
    # Include that growth; otherwise a 768px plan understated the observed peak.
    tile_factor = max(1.0, area / (512 * 512))
    warp = int(pixels * (480 if debayer == "rcd" else 128) * tile_factor) + 64 * area
    # isfinite(st) (three channels), its channel reduction, and output maps.
    coverage = busiest * area * 4 + 24 * area
    # Keep room for useful sort chunks, rather than merely fitting the minimum
    # 4096-column fallback. Integration can lower its chunk further if necessary.
    # feathering adds the gathered weights and their product with the light weights
    integration = busiest * min(32768, area * 3) * (INTEGRATION_BYTES[rejection] + (8 if feather else 0))
    scratch = max(warp, coverage, integration)
    return {"buffer_bytes": buffer, "scratch_bytes": scratch, "required_bytes": buffer + scratch + RESERVE}


def render_same_tile(bounds, render, oom_type, release, retries):
    """Retry exactly once, outside the exception scope, without changing arithmetic geometry."""
    try:
        result = render(bounds)
    except oom_type:
        pass
    else:
        yield bounds, result
        return
    retries.append(list(bounds))
    release(bounds)
    yield bounds, render(bounds)  # second failure propagates; no partial tile is committed
