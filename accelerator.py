import numpy as np
import time
import os
import serial

USE_FPGA = False
USE_FPGA_IN_LOOP = False

RESPONSE_SIZE = 68 # expected responce size of fpga output

START_BYTE = 0xA5
SER_PORT = os.environ.get("FPGA_PORT", "/dev/ttyUSB0")
BAUD = int(os.environ.get("FPGA_BAUD", "115200"))
TIMEOUT_S = 0.5

_ser = None


# Timing log for current token
_timing_log = {
    "quantization_s": 0.0,
    "uart_send_s": 0.0,
    "fpga_compute_s": 0.0,
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
        _timing_log[key] = 0.0
    _timing_log("fpga_cylces") = 0




def quantize_to_int8(x):
    """
    Turns float numbers into small integers (int8)
    Used for the UART protocol which sends bytes (will send 1 byte per number)
    """
    max_abs = np.max(np.abs(x))
    if max_abs < 1e9:
        scale = 1.0
    else:
        scale = 127.0 / max_abs

    x_scaled = np.round(x*scale)
    x_clipped = np.clip(x_scaled, -128, 127)
    return x_clipped.astype(np.int8), scale

def dequantize_from_int32(c_int32, scale_A, scale_B):
    """
    Convert int32 accumulator back to float.
    Since we computed (A * scale_A) @ (B * scale_B) = C * (scale_A * scale_B)
    We divide by both scales to recover the original range.
    """
    return c_int32.astype(np.float64) / (scale_A * scale_B)
def pack_request(A, B):
    """
    Builds the exact bytes being sent to the UART
    
    :param A: 4x4 int8
    :param B: 4x4 int8
    """
    header = bytes([START_BYTE])
    payload = A.tobytes() + B.tobytes()
    return header + payload # 32 + 1 bytes in total


def get_serial():
    """
    Open the serial port once and reuse it
    """
    global _ser
    if _ser is None:
        _ser = serial.Serial(SER_PORT, BAUD, timeout=TIMEOUT_S)
        time.sleep(0.05) # some boards reset when port opens
        _ser.reset_input_buffer()
        _ser.reset_output_buffer()
    return _ser


def uart_send(packet_bytes):
    """
    Send bytes to FPGA over UART
    """
    ser = get_serial()
    ser.write(packet_bytes)
    ser.flush()


def uart_read_result():
    """
    Read 64 bytes from FPGA
    Interpret them as 16 int32 numbers (4x4)
    """
    ser = get_serial()
    data = ser.read(64)
    if len(data) != 64:
        raise TimeoutError(f"Expected 64 bytes but got {len(data)}")

    C = np.frombuffer(data, dtype="<i4").reshape(4,4)
    return C


def fpga_gemm_tile(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Quantises the float values into int8, packs them into bytes,
    sends to FPGA, reads int32 result back
    
    :param A: float 4x4 matrices from the model (token inputs)
    :param B: float 4x4 matrices from the model (weight inputs)
    :return: int32 4x4 matrix
    """
    if not USE_FPGA:
        return cpu_reference_int8(A,B)
    
    A_int8 = quantize_to_int8(A)
    B_int8 = quantize_to_int8(B)
    packet = pack_request(A_int8, B_int8)

    # clear old junk bytes before sending
    ser = get_serial()
    ser.reset_input_buffer()
    uart_send(packet)

    try:
        C_int32 = uart_read_result()
    except TimeoutError:
        print("FPGA timeout, falling back to CPU")
        return cpu_reference_int8(A,B)
   
    return C_int32


def cpu_reference_int8(A, B):
    """
    Does the same math on CPU that FPGA is doing for comparison
    """
    A_int8 = quantize_to_int8(A)
    B_int8 = quantize_to_int8(B)
    return A_int8.astype(np.int32) @ B_int8.astype(np.int32)