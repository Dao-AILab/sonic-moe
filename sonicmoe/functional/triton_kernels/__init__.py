import torch
import triton
import triton.language as tl

from .bitmatrix import _bitmatrix_metadata_compute_stage1, _bitmatrix_metadata_compute_stage2


@triton.jit
def _compute_col_partial_sum_kernel(
    topk_indices_ptr,
    partial_sum_ptr,
    T,
    E: tl.constexpr,
    n_tiles,
    TOKENS_PER_TILE: tl.constexpr,
    K_POW2: tl.constexpr,  # next_power_of_2(K),
    K: tl.constexpr,  # actual number of experts per token
    E_POW2: tl.constexpr,  # next_power_of_2(E)
):
    # One CTA per tile. Tile `t` covers tokens [t * TOKENS_PER_TILE, (t+1) * TOKENS_PER_TILE).
    # Produces partial_sum[e, tile_id] = number of entries in this tile routed to expert e.
    # Layout: partial_sum is [E, n_tiles] (row-major), so partial_sum[e, t] = partial_sum_ptr + e * n_tiles + t.
    # Caller transposes to [n_tiles, E] before passing to stage1/stage2.
    tile_id = tl.program_id(0)

    # Zero this tile's column in partial_sum[*, tile_id].
    # Chunked by E_POW2 to keep vector width a power of 2.
    for e_start in tl.static_range(0, E, E_POW2):
        e_offs = e_start + tl.arange(0, E_POW2)
        tl.store(
            partial_sum_ptr + e_offs * n_tiles + tile_id,
            tl.zeros([E_POW2], tl.int32),
            mask=e_offs < E,
        )

    # Load expert ids for this tile: shape [TOKENS_PER_TILE, K_POW2].
    # Tokens beyond T and k-slots beyond K are masked out (other=-1).
    tok_offs = tile_id * TOKENS_PER_TILE + tl.arange(0, TOKENS_PER_TILE)
    k_offs = tl.arange(0, K_POW2)
    tok_mask = tok_offs < T

    load_mask = tok_mask[:, None] & (k_offs[None, :] < K)
    safe_k = tl.minimum(k_offs, K - 1)  # avoid OOB when k_offs >= K
    expert_ids = tl.load(
        topk_indices_ptr + tok_offs[:, None] * K + safe_k[None, :],
        mask=load_mask,
        other=-1,
    )

    # Flatten to [TOKENS_PER_TILE * K_POW2] and histogram into partial_sum.
    # safe_experts remaps masked (-1) entries to expert 0 (harmless: flat_mask=False).
    flat_experts = tl.reshape(expert_ids, [TOKENS_PER_TILE * K_POW2])
    flat_mask = tl.reshape(load_mask, [TOKENS_PER_TILE * K_POW2])
    safe_experts = tl.where(flat_mask, flat_experts, 0)

    tl.atomic_add(
        partial_sum_ptr + safe_experts * n_tiles + tile_id,
        tl.full([TOKENS_PER_TILE * K_POW2], 1, dtype=tl.int32),
        mask=flat_mask,
    )


def TC_topk_router_metadata_triton(
    topk_router_indices: torch.Tensor, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T, K = topk_router_indices.size()
    TK = T * K
    device = topk_router_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = 1024 // K_POW2
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # ── Kernel 1: tiled histogram ─────────────────────────────────────────────
    # col_partial_sum_trans[E, n_tiles]: raw per-expert-per-tile counts.
    # Stored transposed so each CTA writes to its own column (tile_id), avoiding
    # cross-CTA write conflicts. Transposed back to [n_tiles, E] for stage1/stage2.
    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _compute_col_partial_sum_kernel[(n_tiles,)](
        topk_router_indices,
        col_partial_sum_trans,
        T,
        E,
        n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )
    expert_frequency = col_partial_sum_trans.sum(dim=1, dtype=torch.int32)
    col_partial_sum = col_partial_sum_trans.T  # [n_tiles, E]

    # ── Kernel 2: stage1 ─────────────────────────────────────────────────────
    # - For each expert e (pid < E): convert col_partial_sum[*, e] from raw
    #   counts to exclusive prefix sums over tiles in-place.
    # - For pid == E: write exclusive cumsum of expert_freq_offset into
    #   expert_freq_off[0:E] (= col_offs, a view into expert_freq_off).
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    _bitmatrix_metadata_compute_stage1[(E + 2,)](
        expert_frequency,
        expert_frequency_offset,
        E,
        col_partial_sum,
        n_tiles,
        TK,
        BLOCK_M=128,
        BLOCK_N=E_POW2,
    )

    # ── Kernel 3: stage2 ─────────────────────────────────────────────────────
    # For each tile: sort entries by expert, compute output positions, scatter.
    _bitmatrix_metadata_compute_stage2[(n_tiles,)](
        s_scatter_idx,
        s_reverse_scatter_idx,
        x_gather_idx,
        topk_router_indices,
        T,
        col_partial_sum,
        n_tiles,
        expert_frequency_offset[:E],
        K_POW2=K_POW2,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        K=K,
    )

    return (expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx)
