# Mac Studio + DGX Spark Flash cluster notes

Working tree patches for `orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX` on Metal + CUDA MLX ring.

## In-repo (this branch)

- `EXO_METAL_RANK0` — Metal before Cuda in Pipeline cycle (avoids Spark CUDA OOM on early-layer packs)
- `EXO_MAX_LAYER_FRAC` / layer-cap rebalance
- `EXO_CAT6_PREF` — prefer `10.10.10.x` over Wi-Fi for ring hosts
- Smart download allow-patterns (root 4-bit only)
- anyio 4.15 `MemoryObjectStreamState` import fix
- `EXO_SHARD_BEFORE_EVAL` — drop non-shard layers before `mx.eval`
- `EXO_Qwen4_HYBRID_FA_IDX` — recompute `fa_idx`/`ssm_idx` for mlx_vlm qwen4_exp shards
- Pipeline send/recv: `mx.synchronize()` + shape logs (Metal↔CUDA fence stall debug)

## Out-of-repo (apply on both Mac + Spark venvs)

`mlx_vlm` `qwen3_5/language.py` — ArraysCache has no `.offset`; guard `_idx` / `offset` before use (same edit on Mac and Spark site-packages).

## Local helpers (not committed)

- `~/bin/exo-mac-start.sh` — `--legacy-daemon --offline --no-downloads -m --namespace cryptobandit-spark-mac`
- `~/bin/exo-spark-start.sh` — `--offline --no-downloads --namespace cryptobandit-spark-mac`
- `~/bin/ssh-spark-cat6` — always targets Host `SparkCat6` (`10.10.10.2`)

## Status

LoadModel with Mac rank0 / Spark rank1 works. First warmup prefill over Metal↔CUDA ring still under debug (was Fence::wait / 0% GPU spin before synchronize patch).

## Prefill / PLE notes (2026-09-15)

- Short pipeline warmup uses **serial stages** (rank0 forward+queue, barrier, then rank0 flush concurrent with rank1 recv) so Qwen4 PLE `mx.eval` mid-forward does not Fence-deadlock against a posted PP recv, and flush does not deadlock waiting for that recv.
- PLE ngram weights are ~**111GB** on disk (`ShardedEmbedding`, `split_ngram_parts=128`). Do **not** `mx.eval` those shard params at load — leave them lazy and gather touched rows only.
- After a thrash, macOS may leave **~100GB+ wired Metal** with no process holding it; reboot (or long wait) to reclaim before `place_instance` can find cycles.
