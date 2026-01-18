#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from accelerator import fpga_gemm_tile

# --------------------------
# Quantization: scale = 127 / max_abs; q = round(x*scale) clipped to int8
# --------------------------
def quantize_symmetric_int8(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    max_abs = float(np.max(np.abs(x)))
    scale = 1.0 if max_abs == 0.0 else (127.0 / max_abs)
    q = np.rint(x * scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale

def to_i32_4x4(flat16):
    return np.array(flat16, dtype=np.int32).reshape(4, 4)

def is_int32_tile_candidate(obj) -> bool:
    try:
        arr = np.asarray(obj)
    except Exception:
        return False
    if arr.size != 16:
        return False
    if np.issubdtype(arr.dtype, np.integer):
        return True
    if arr.dtype == object:
        try:
            _ = np.array([int(v) for v in arr.flatten()], dtype=np.int32)
            return True
        except Exception:
            return False
    return False

def extract_fpga_tile_and_cycles(ret):
    C_flat = None
    cycles = None

    if isinstance(ret, dict):
        for k in ("C", "c", "c_int32", "c_out", "tile", "result"):
            if k in ret and is_int32_tile_candidate(ret[k]):
                C_flat = ret[k]
                break
        for k in ("cycles", "fpga_cycles", "cycle_count"):
            if k in ret:
                try:
                    cycles = int(ret[k])
                except Exception:
                    cycles = None
                break
        if C_flat is None:
            raise RuntimeError(f"Could not find int32 tile in dict return. keys={list(ret.keys())}")
        return C_flat, cycles

    if isinstance(ret, (tuple, list)):
        for item in ret:
            if is_int32_tile_candidate(item):
                C_flat = item
                break

        for item in ret:
            if isinstance(item, (int, np.integer)):
                v = int(item)
                if 1 <= v <= 1_000_000_000 and v not in (0x5A, 0x5B, 0xA6, 0xA5):
                    if v == 10:
                        cycles = v
                        break
                    if cycles is None:
                        cycles = v

        if C_flat is None:
            raise RuntimeError(f"Could not locate a 16-int32 tile in fpga_gemm_tile return.\nReturn={ret}")

        return C_flat, cycles

    raise RuntimeError(f"Unsupported fpga_gemm_tile return type: {type(ret)}")

def add_matrix_table(ax, title, M):
    """Render a 4x4 int matrix as a clean table in a matplotlib axis."""
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=8)

    cell_text = [[f"{int(M[r,c])}" for c in range(4)] for r in range(4)]
    row_labels = [f"r{r}" for r in range(4)]
    col_labels = [f"c{c}" for c in range(4)]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)  # make it readable

def main():
    rng = np.random.default_rng(0)

    # Float inputs (like your pipeline)
    A_f = rng.standard_normal((4, 4), dtype=np.float32)
    B_f = rng.standard_normal((4, 4), dtype=np.float32)

    # CPU reference on the actual quantized int8 values that get sent
    A_q, sA = quantize_symmetric_int8(A_f)
    B_q, sB = quantize_symmetric_int8(B_f)
    C_cpu = (A_q.astype(np.int32) @ B_q.astype(np.int32)).astype(np.int32)

    # FPGA result from your existing stack
    ret = fpga_gemm_tile(A_f, B_f)
    C_fpga_flat, cycles = extract_fpga_tile_and_cycles(ret)
    C_fpga = to_i32_4x4(C_fpga_flat)

    diff = (C_fpga - C_cpu).astype(np.int32)
    match = np.array_equal(C_fpga, C_cpu)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")

    status = "MATCH ✅" if match else "MISMATCH ❌"
    cyc_str = f"{cycles}" if cycles is not None else "?"
    ax_title.text(
        0.5, 0.72,
        f"FPGA vs CPU Correctness Check: {status}",
        ha="center", va="center", fontsize=16
    )
    ax_title.text(
        0.5, 0.35,
        f"Quant scales: sA={sA:.3f}, sB={sB:.3f}    |    fpga_cycles={cyc_str}",
        ha="center", va="center", fontsize=11
    )

    ax_cpu  = fig.add_subplot(gs[1, 0])
    ax_fpga = fig.add_subplot(gs[1, 1])
    ax_diff = fig.add_subplot(gs[1, 2])

    add_matrix_table(ax_cpu,  "C_cpu (int32) on quantized int8", C_cpu)
    add_matrix_table(ax_fpga, "C_fpga (int32) returned over UART", C_fpga)
    add_matrix_table(ax_diff, "diff = fpga − cpu", diff)

    fig.tight_layout()
    out = "correctness_compare.png"
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
