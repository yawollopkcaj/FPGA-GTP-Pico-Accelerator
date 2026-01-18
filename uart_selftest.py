import numpy as np
from accelerator import fpga_gemm_tile, cpu_reference_int8

A = np.array([
    [0.1, 0.2, -0.1, 0.0],
    [0.0, -0.3, 0.2, 0.1],
    [0.5, 0.0, 0.0, -0.2],
    [0.1, 0.1, 0.1, 0.1],
], dtype=np.float32)

B = np.array([
    [0.2, 0.0, 0.1, -0.1],
    [0.0, 0.3, 0.0, 0.2],
    [0.1, 0.0, -0.2, 0.0],
    [0.0, 0.1, 0.0, 0.4],
], dtype=np.float32)

C_fpga = fpga_gemm_tile(A, B)
C_cpu  = cpu_reference_int8(A, B)

print("FPGA:\n", C_fpga)
print("CPU:\n", C_cpu)
print("MATCH?", np.array_equal(C_fpga, C_cpu))