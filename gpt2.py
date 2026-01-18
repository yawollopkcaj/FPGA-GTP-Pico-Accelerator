import numpy as np
import time
from accelerator import (
    fpga_gemm_tile,
    dequantize_from_int32,
    verify_fpga_result,
    get_timing_log,
    reset_timing_log,
    _timing_log,
    close_serial,
)

TILE_M = 4  # how many rows of tokens we take
TILE_K = 4  # how many input features we take
TILE_N = 4  # how many output features we take

token_timing_history = []

fpga_shadow_pending = False
fpga_shadow_A = None
fpga_shadow_B = None

_token_cpu_linear_s = 0.0
_token_linear_calls = 0


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)
    return g * x + b


def linear(x, w, b):
    """
    Linear layer: y = x @ w + b

    FPGA mode: ONCE PER TOKEN (shadow compute)
      - We DO NOT call FPGA here (to avoid 48x UART round-trips per token).
      - We only:
          * compute CPU output normally (used for correctness)
          * accumulate per-token CPU linear time + call counts
          * stash one representative 4x4 tile (A,B) the first time per token
            so generate() can send exactly one transaction to FPGA.
    """
    import accelerator
    global _timing_log
    global _token_cpu_linear_s, _token_linear_calls
    global fpga_shadow_pending, fpga_shadow_A, fpga_shadow_B
    global _token_linear_macs

    _token_linear_calls += 1

    # Count MACs for this GEMM
    # x @ w where x is (..., K) and w is (K, N)
    try:
        if x.ndim == 2:
            M = int(x.shape[0])
            K = int(x.shape[1])
        elif x.ndim == 3:
            M = int(x.shape[1])
            K = int(x.shape[2])
        else:
            M = 0
            K = 0
        N = int(w.shape[1])
        _token_linear_macs += (M * K * N)
    except Exception:
        pass

    t0 = time.perf_counter()
    y = x @ w
    if b is not None:
        y = y + b
    t1 = time.perf_counter()

    dt = (t1 - t0)
    _token_cpu_linear_s += dt
    _timing_log["cpu_remaining_s"] += dt

    if not accelerator.USE_FPGA:
        _timing_log["total_s"] += dt
        return y

    if not fpga_shadow_pending:
        try:
            if x.ndim == 2:
                A = np.asarray(x[:TILE_M, :TILE_K], dtype=np.float32, order="C")
            elif x.ndim == 3:
                A = np.asarray(x[0, :TILE_M, :TILE_K], dtype=np.float32, order="C")
            else:
                A = None

            if A is not None:
                B_tile = np.asarray(w[:TILE_K, :TILE_N], dtype=np.float32, order="C")
                fpga_shadow_A = A
                fpga_shadow_B = B_tile
                fpga_shadow_pending = True
        except Exception:
            pass

    return y



def log_token_timing(token_id=None):
    """Save timing data for this token."""
    timing = get_timing_log()
    timing["token_id"] = token_id
    timing["timestamp"] = time.time()
    token_timing_history.append(timing)
    reset_timing_log()
    return timing


def print_timing_summary(timing=None):
    """Pretty print a timing breakdown."""
    if timing is None:
        timing = get_timing_log()

    print("\n" + "=" * 50)
    print("TIMING BREAKDOWN")
    print("=" * 50)
    print(f"  Quantization:      {timing['quantization_s']*1000:8.3f} ms")
    print(f"  UART Send:         {timing['uart_send_s']*1000:8.3f} ms")
    print(f"  UART Receive:      {timing.get('uart_recv_s', 0)*1000:8.3f} ms")
    print(f"  Dequantization:    {timing['dequantization_s']*1000:8.3f} ms")
    print(f"  CPU Remaining:     {timing['cpu_remaining_s']*1000:8.3f} ms")
    print("-" * 50)
    print(f"  TOTAL:             {timing['total_s']*1000:8.3f} ms")
    print("-" * 50)
    print(f"  FPGA Cycles:       {timing['fpga_cycles']:>8d}")
    print(f"  Linear Calls:      {timing.get('linear_calls', 'N/A'):>8}")
    print("=" * 50 + "\n")


def save_timing_history_csv(filepath="timing_log.csv"):
    """Save all token timing history to CSV."""
    import csv

    if not token_timing_history:
        print("No timing data to save.")
        return

    fieldnames = [
    "token_id", "timestamp", "total_s",
    "quantization_s", "uart_send_s", "uart_recv_s",
    "dequantization_s", "cpu_remaining_s",
    "fpga_cycles", "scale_A", "scale_B",

    "cpu_linear_s", "linear_calls", "token_linear_macs",

    "tile_cpu_float_4x4_s", "tile_cpu_int8_4x4_s",
    "tile_uart_wall_s", "tile_ideal_uart_s",
    "tile_fpga_compute_est_s", "tile_uart_overhead_est_s",

    "fpga_macs_per_cycle_eff", "fpga_eff_gmacs",
    ]


    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in token_timing_history:
            writer.writerow(row)

    print(f"Saved {len(token_timing_history)} timing records to {filepath}")


# ============================================================
# GPT-2 MODEL
# ============================================================
def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv = np.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = np.hstack(out_heads)
    x = linear(x, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    import accelerator

    global _token_cpu_linear_s, _token_linear_calls, _token_linear_macs
    global fpga_shadow_pending, fpga_shadow_A, fpga_shadow_B
    global _timing_log

    if "_token_cpu_linear_s" not in globals():
        _token_cpu_linear_s = 0.0
    if "_token_linear_calls" not in globals():
        _token_linear_calls = 0
    if "_token_linear_macs" not in globals():
        _token_linear_macs = 0

    print(f"\n{'='*60}")
    print(f"Generating {n_tokens_to_generate} tokens")
    print(f"USE_FPGA: {accelerator.USE_FPGA}")
    print("FPGA MODE: once per token (shadow compute)")
    print(f"{'='*60}\n")

    for i in tqdm(range(n_tokens_to_generate), "generating"):
        reset_timing_log()
        _token_cpu_linear_s = 0.0
        _token_linear_calls = 0
        _token_linear_macs = 0

        fpga_shadow_pending = False
        fpga_shadow_A = None
        fpga_shadow_B = None

        t_token0 = time.perf_counter()

        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))

        fpga_cycles = 0
        fpga_stats = None

        if accelerator.USE_FPGA and fpga_shadow_pending and (fpga_shadow_A is not None) and (fpga_shadow_B is not None):
            try:
                c_fpga, scale_A, scale_B, fpga_cycles, fpga_stats = fpga_gemm_tile(fpga_shadow_A, fpga_shadow_B)
                matches, max_err = verify_fpga_result(fpga_shadow_A, fpga_shadow_B, c_fpga, scale_A, scale_B)
                if not matches:
                    print(f"⚠ FPGA mismatch on shadow tile! Max error: {max_err}")
            except Exception as e:
                print(f"FPGA shadow compute failed: {e}")
                fpga_cycles = 0
                fpga_stats = None

        t_token1 = time.perf_counter()
        token_wall_s = (t_token1 - t_token0)

        _timing_log["total_s"] += token_wall_s
        _timing_log["fpga_cycles"] = int(fpga_cycles)

        _timing_log["cpu_linear_s"] = float(_token_cpu_linear_s)
        _timing_log["linear_calls"] = int(_token_linear_calls)
        _timing_log["token_linear_macs"] = int(_token_linear_macs)

        if fpga_stats is not None:
            _timing_log["tile_cpu_float_4x4_s"] = float(fpga_stats.get("cpu_float_4x4_s", 0.0))
            _timing_log["tile_cpu_int8_4x4_s"]  = float(fpga_stats.get("cpu_int8_4x4_s", 0.0))
            _timing_log["tile_uart_wall_s"]     = float(fpga_stats.get("uart_wall_s", 0.0))
            _timing_log["tile_ideal_uart_s"]    = float(fpga_stats.get("ideal_uart_s", 0.0))
            _timing_log["tile_fpga_compute_est_s"] = float(fpga_stats.get("fpga_compute_est_s", 0.0))
            _timing_log["tile_uart_overhead_est_s"] = float(fpga_stats.get("uart_overhead_est_s", 0.0))

            # MAC-scaled estimate for "FPGA does all linear GEMMs, no UART"
            # macs_per_cycle_eff ~ 64 / fpga_cycles_for_tile
            # fpga_time ~ token_macs / (macs_per_cycle_eff * FPGA_CLK_HZ)
            try:
                fpga_cycles_tile = int(fpga_stats.get("fpga_cycles", 0))
                if fpga_cycles_tile > 0:
                    macs_per_cycle_eff = 64.0 / float(fpga_cycles_tile)
                    fpga_gmacs = (macs_per_cycle_eff * float(accelerator.FPGA_CLK_HZ)) / 1e9
                    fpga_linear_no_uart_s = float(_token_linear_macs) / (macs_per_cycle_eff * float(accelerator.FPGA_CLK_HZ))

                    _timing_log["fpga_macs_per_cycle_eff"] = macs_per_cycle_eff
                    _timing_log["fpga_eff_gmacs"] = fpga_gmacs
                    _timing_log["fpga_linear_no_uart_s"] = fpga_linear_no_uart_s
            except Exception:
                pass

        timing = log_token_timing(token_id=int(next_id))

        if accelerator.USE_FPGA and (i < 3 or i == n_tokens_to_generate - 1):
            ms = 1000.0
            print(f"\nToken {i+1}: id={next_id}, total={timing['total_s']*ms:.1f}ms, fpga_cycles={timing['fpga_cycles']}")

            cpu_linear_ms = timing.get("cpu_linear_s", 0.0) * ms
            n_linear = timing.get("linear_calls", 0)
            macs = timing.get("token_linear_macs", 0)
            print(f"  CPU linear total: {cpu_linear_ms:.1f}ms over {n_linear} linear() calls")
            print(f"  Token linear MACs: {macs:,}")

            if fpga_stats is not None:
                cpu_float_us = timing.get("tile_cpu_float_4x4_s", 0.0) * 1e6
                cpu_int8_us  = timing.get("tile_cpu_int8_4x4_s", 0.0) * 1e6
                fpga_comp_ms = timing.get("tile_fpga_compute_est_s", 0.0) * ms
                uart_wall_ms = timing.get("tile_uart_wall_s", 0.0) * ms
                uart_ideal_ms = timing.get("tile_ideal_uart_s", 0.0) * ms

                print(f"  Shadow tile 4x4:")
                print(f"    CPU(float32) ~ {cpu_float_us:.2f} us")
                print(f"    CPU(int8->i32) ~ {cpu_int8_us:.2f} us")
                print(f"    FPGA compute est ~ {fpga_comp_ms:.6f} ms (from cycles/clk)")
                print(f"    UART wall ~ {uart_wall_ms:.1f} ms (ideal line-time ~ {uart_ideal_ms:.2f} ms)")

                fpga_eff = timing.get("fpga_eff_gmacs", 0.0)
                fpga_linear_no_uart_s = timing.get("fpga_linear_no_uart_s", None)
                if fpga_linear_no_uart_s is not None:
                    print(f"  MAC-scaled estimate (compute-only):")
                    print(f"    FPGA effective throughput ~ {fpga_eff:.3f} GMAC/s")
                    print(f"    FPGA time for ALL linear GEMMs (no UART) ~ {fpga_linear_no_uart_s*ms:.1f} ms/token")
                    if timing.get("cpu_linear_s", 0.0) > 0:
                        speedup = timing["cpu_linear_s"] / fpga_linear_no_uart_s if fpga_linear_no_uart_s > 0 else 0.0
                        print(f"    (vs CPU linear) ~ {speedup:.2f}×")

    return inputs[len(inputs) - n_tokens_to_generate:]



def main(prompt: str, n_tokens_to_generate: int = 20, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    import accelerator

    print(f"\n{'#'*60}")
    print("FPGA-Accelerated GPT-2 Demo")
    print(f"{'#'*60}")
    print(f"Model: {model_size}")
    print(f"Prompt: '{prompt}'")
    print(f"Tokens to generate: {n_tokens_to_generate}")

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)

    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    t_start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    t_end = time.time()

    output_text = encoder.decode(output_ids)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated text: {output_text}")
    print(f"Total time: {t_end - t_start:.2f}s")
    print(f"Tokens/sec: {n_tokens_to_generate / (t_end - t_start):.2f}")

    if accelerator.USE_FPGA and token_timing_history:
        save_timing_history_csv("fpga_timing.csv")

    close_serial()

    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)
