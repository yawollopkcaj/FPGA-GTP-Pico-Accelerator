import numpy as np
import time
import os
import serial

# ============================================================
# CONFIGURATION
# ============================================================
USE_FPGA = True                  # Set to True to use FPGA
USE_FPGA_IN_LOOP = False         # Set to True to use FPGA result in model output

START_BYTE = 0xA5
ACK_BYTE = 0x5A
END_BYTE = 0xA6

SER_PORT = os.environ.get("FPGA_PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("FPGA_BAUD", "115200"))
TIMEOUT_S = 2.0  # Longer timeout for safety

# --- Performance model knobs ---
FPGA_CLK_HZ = int(os.environ.get("FPGA_CLK_HZ", "50000000"))  # DE0-CV clock driving your systolic core
BITS_PER_UART_BYTE = 10  # 1 start + 8 data + 1 stop (8N1)

_ser = None

# ============================================================
# TIMING LOG
# ============================================================
_timing_log = {
    "quantization_s": 0.0,
    "uart_send_s": 0.0,
    "uart_recv_s": 0.0,
    "dequantization_s": 0.0,
    "cpu_remaining_s": 0.0,
    "total_s": 0.0,
    "fpga_cycles": 0,
    "scale_A": 0.0,
    "scale_B": 0.0,
}


def get_timing_log():
    return _timing_log.copy()


def reset_timing_log():
    global _timing_log
    for key in _timing_log:
        if isinstance(_timing_log[key], float):
            _timing_log[key] = 0.0
        else:
            _timing_log[key] = 0


# ============================================================
# QUANTIZATION
# ============================================================
def quantize_to_int8(x):
    """
    Dynamic per-tensor quantization: scales based on actual data range.
    Returns (int8 array, scale factor used)
    """
    max_abs = np.max(np.abs(x))
    if max_abs < 1e-9:
        scale = 1.0
    else:
        scale = 127.0 / max_abs

    x_scaled = np.round(x * scale)
    x_clipped = np.clip(x_scaled, -128, 127)
    return x_clipped.astype(np.int8), scale


def dequantize_from_int32(c_int32, scale_A, scale_B):
    """
    Convert int32 accumulator back to float.
    Since we computed (A * scale_A) @ (B * scale_B) = C * (scale_A * scale_B)
    We divide by both scales to recover the original range.
    """
    return c_int32.astype(np.float64) / (scale_A * scale_B)


# ============================================================
# UART PROTOCOL
# ============================================================
def pack_request(A_int8, B_int8):
    """
    Builds the packet to send to FPGA.
    Protocol: [0xA5] [A bytes (16)] [B bytes (16)]
    Total: 33 bytes
    """
    header = bytes([START_BYTE])
    payload = A_int8.tobytes() + B_int8.tobytes()
    return header + payload


def get_serial():
    """Open the serial port once and reuse it."""
    global _ser
    if _ser is None:
        _ser = serial.Serial(SER_PORT, BAUD, timeout=TIMEOUT_S)
        time.sleep(0.1)  # Let FPGA settle after port open
        _ser.reset_input_buffer()
        _ser.reset_output_buffer()
    return _ser


def close_serial():
    """Close the serial port."""
    global _ser
    if _ser is not None:
        _ser.close()
        _ser = None


def uart_transact(packet_bytes):
    """
    Send packet to FPGA and receive response.

    Protocol:
      Send: [0xA5] [A: 16 bytes] [B: 16 bytes] = 33 bytes
      Recv: [0x5A] [C: 64 bytes] [cycles: 4 bytes] [0xA6] = 70 bytes

    Returns (C_int32 4x4, cycle_count) or raises exception on error.
    """
    ser = get_serial()
    ser.reset_input_buffer()

    ser.write(packet_bytes)
    ser.flush()

    ack = ser.read(1)
    if len(ack) != 1:
        raise TimeoutError("No ACK received from FPGA")
    if ack[0] != ACK_BYTE:
        raise ValueError(f"Bad ACK byte: 0x{ack[0]:02X}, expected 0x{ACK_BYTE:02X}")

    data = ser.read(68)
    if len(data) != 68:
        raise TimeoutError(f"Expected 68 bytes, got {len(data)}")

    end = ser.read(1)
    if len(end) != 1 or end[0] != END_BYTE:
        raise ValueError(f"Bad END marker: {end.hex() if end else 'none'}")

    C_int32 = np.frombuffer(data[:64], dtype="<i4").reshape(4, 4)
    cycle_count = np.frombuffer(data[64:68], dtype="<u4")[0]

    return C_int32, cycle_count


# ============================================================
# CPU REFERENCE (for verification and fallback)
# ============================================================
def cpu_reference_int8(A, B):
    """
    Does the same int8 math on CPU that FPGA does.
    Returns (C_int32, scale_A, scale_B)
    """
    A_int8, scale_A = quantize_to_int8(A)
    B_int8, scale_B = quantize_to_int8(B)
    C_int32 = A_int8.astype(np.int32) @ B_int8.astype(np.int32)
    return C_int32, scale_A, scale_B


def verify_fpga_result(A, B, c_fpga, scale_A, scale_B):
    """
    Compare FPGA int32 result to CPU int32 reference.
    Returns (matches: bool, max_error: int)
    """
    A_q = np.clip(np.round(A * scale_A), -128, 127).astype(np.int8)
    B_q = np.clip(np.round(B * scale_B), -128, 127).astype(np.int8)
    c_cpu = A_q.astype(np.int32) @ B_q.astype(np.int32)

    diff = c_cpu - c_fpga
    max_err = int(np.max(np.abs(diff)))
    return (max_err == 0), max_err


def estimate_uart_time_s(num_bytes: int) -> float:
    return (num_bytes * BITS_PER_UART_BYTE) / BAUD


def estimate_fpga_compute_time_s(fpga_cycles: int) -> float:
    return fpga_cycles / float(FPGA_CLK_HZ)


def cpu_benchmark_float_matmul(A: np.ndarray, B: np.ndarray, iters: int = 2000) -> float:
    Af = np.asarray(A, dtype=np.float32)
    Bf = np.asarray(B, dtype=np.float32)

    _ = Af @ Bf

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = Af @ Bf
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def cpu_benchmark_int8_matmul(A_int8: np.ndarray, B_int8: np.ndarray, iters: int = 20000) -> float:
    Ai = np.asarray(A_int8, dtype=np.int8)
    Bi = np.asarray(B_int8, dtype=np.int8)

    _ = Ai.astype(np.int32) @ Bi.astype(np.int32)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = Ai.astype(np.int32) @ Bi.astype(np.int32)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# ============================================================
# MAIN FPGA GEMM FUNCTION
# ============================================================
def fpga_gemm_tile(A: np.ndarray, B: np.ndarray):
    """
    Compute a 4x4 GEMM tile on FPGA.

    Args:
        A: 4x4 float array (activations)
        B: 4x4 float array (weights)

    Returns:
        (C_int32, scale_A, scale_B, fpga_cycles)

    If USE_FPGA is False or FPGA fails, falls back to CPU reference.
    """
    global _timing_log

    t_quant_start = time.perf_counter()
    A_int8, scale_A = quantize_to_int8(A)
    B_int8, scale_B = quantize_to_int8(B)
    packet = pack_request(A_int8, B_int8)
    t_quant_end = time.perf_counter()
    _timing_log["quantization_s"] += t_quant_end - t_quant_start
    _timing_log["scale_A"] = scale_A
    _timing_log["scale_B"] = scale_B

    cpu_float_s = cpu_benchmark_float_matmul(A, B, iters=2000)
    cpu_int8_s = cpu_benchmark_int8_matmul(A_int8, B_int8, iters=20000)
    ideal_uart_s = estimate_uart_time_s(33 + 70)

    if not USE_FPGA:
        C_int32 = A_int8.astype(np.int32) @ B_int8.astype(np.int32)
        stats = {
            "uart_wall_s": 0.0,
            "ideal_uart_s": ideal_uart_s,
            "fpga_cycles": 0,
            "fpga_compute_est_s": 0.0,
            "uart_overhead_est_s": 0.0,
            "cpu_float_4x4_s": cpu_float_s,
            "cpu_int8_4x4_s": cpu_int8_s,
        }
        return C_int32, scale_A, scale_B, 0, stats

    try:
        t_uart0 = time.perf_counter()
        C_int32, fpga_cycles = uart_transact(packet)
        t_uart1 = time.perf_counter()

        uart_wall_s = t_uart1 - t_uart0
        _timing_log["uart_recv_s"] += uart_wall_s
        _timing_log["fpga_cycles"] = int(fpga_cycles)

        fpga_compute_est_s = estimate_fpga_compute_time_s(int(fpga_cycles))
        uart_overhead_est_s = max(0.0, uart_wall_s - fpga_compute_est_s)

        stats = {
            "uart_wall_s": uart_wall_s,
            "ideal_uart_s": ideal_uart_s,
            "fpga_cycles": int(fpga_cycles),
            "fpga_compute_est_s": fpga_compute_est_s,
            "uart_overhead_est_s": uart_overhead_est_s,
            "cpu_float_4x4_s": cpu_float_s,
            "cpu_int8_4x4_s": cpu_int8_s,
        }

        return C_int32, scale_A, scale_B, int(fpga_cycles), stats

    except Exception as e:
        print(f"FPGA error: {e}, falling back to CPU")
        C_int32 = A_int8.astype(np.int32) @ B_int8.astype(np.int32)

        stats = {
            "uart_wall_s": 0.0,
            "ideal_uart_s": ideal_uart_s,
            "fpga_cycles": 0,
            "fpga_compute_est_s": 0.0,
            "uart_overhead_est_s": 0.0,
            "cpu_float_4x4_s": cpu_float_s,
            "cpu_int8_4x4_s": cpu_int8_s,
        }

        return C_int32, scale_A, scale_B, 0, stats


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FPGA Accelerator Test")
    print("=" * 60)
    print(f"USE_FPGA: {USE_FPGA}")
    print(f"Port: {SER_PORT}")
    print()

    np.random.seed(42)
    A = np.random.randn(4, 4).astype(np.float32)
    B = np.random.randn(4, 4).astype(np.float32)

    print("A =")
    print(A)
    print("\nB =")
    print(B)

    reset_timing_log()
    C_int32, scale_A, scale_B, cycles, stats = fpga_gemm_tile(A, B)

    matches, max_err = verify_fpga_result(A, B, C_int32, scale_A, scale_B)

    print(f"\nFPGA cycles: {cycles}")
    print(f"Scales: A={scale_A:.3f}, B={scale_B:.3f}")
    print(f"Match: {matches}, Max error: {max_err}")

    C_float = dequantize_from_int32(C_int32, scale_A, scale_B)
    C_cpu = A @ B

    print(f"\nFPGA result (dequantized):")
    print(C_float)
    print(f"\nCPU result (float):")
    print(C_cpu)
    print(f"\nMax float difference: {np.max(np.abs(C_float - C_cpu)):.6f}")

    timing = get_timing_log()
    print(f"\nTiming:")
    print(f"  Quantization: {timing['quantization_s']*1000:.3f} ms")
    print(f"  UART total:   {timing.get('uart_recv_s', 0)*1000:.3f} ms")

    close_serial()
    print("\nDone.")
