# Project Q-Tensor: Hardware Acceleration for LLM Inference

## Summary

**Project Q-Tensor** is a high-performance, energy-efficient hybrid inference engine designed to accelerate Large Language Model (LLM) operations using FPGA-based **systolic arrays**. By offloading computationally intensive General Matrix-Matrix Multiplication (GEMM) tasks from a host CPU to custom RTL, this project addresses the "Von Neumann Tax" (the performance bottleneck caused by constant data movement between the processor and main memory).

## The Problem We Set Out to Solve

LLM inference is dominated by matrix multiplication (attention and linear layers). On conventional CPUs, each multiply-accumulate step requires fetching instructions and operands from memory, creating significant overhead that limits throughput and wastes energy. The CPU is idle (waiting for memory transfer) more than actively computing.

## Solution Highlights

<p align="center">
  <sub><i>Data Movement on the Systolic Array and FPGA Plane in Operation</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/53bc1dd6-7ad0-4c98-80dd-dd503108c81f" width="600" />
</p>

* **Custom Systolic Array:** Developed a scalable systolic array architecture in Verilog to maximize data reuse and achieve 100% compute utilization during matrix operations.
* **Hybrid Inference Engine:** Engineered a system that allows for a hybrid backend using Int8 quantization to offload MAC operations.
* **Energy Efficiency:** Achieved a **2.2X reduction in energy consumption** per $4 \times 4$ GEMM tile compared to traditional CPU execution baseline (analysis done with Quartus Prime Power Analyzer).
* **Performance:** Demonstrated a hardware compute-only speedup reaching an effective throughput of **0.320 GMAC/s**.
* **Power Savings:** Realized a saving of approximately **65 mW per $4 \times 4$ tile** compared to CPU-based compute.

## Results

| Metric | CPU (Fallback) | FPGA Accelerator (Hybrid) |
| :--- | :--- | :--- |
| **Energy per $4 \times 4$ GEMM Tile** | $\sim 26\ nJ$ | $\sim 12\ nJ$ |
| **Compute Time (Micro-benchmark)** | $2.33\ \mu s$ (Int8) | $\sim 0.20\ \mu s$ (Est.) |
| **Utilization** | Sequential / Cache-dependent | 100% Utilization |

<p align="center">
  <sub><i>CPU vs. FPGA runtime benchmark</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d3e0c325-b593-4fb4-9a86-a27e6249023c" width="600" />
</p>

<p align="center">
  <sub><i>Energy per 4x4 GEMM (Quartus Prime Power Analyzer)</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/4f12b3a4-f8d4-4676-872f-bebfe1e75780" width="400" />
</p>

<p align="center">
  <sub><i>Proof of the Primary Bottleneck in the Current Implementation</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/472b30b5-ede4-4b56-82b5-5a0ad672b63b" width="600" />
</p>

*Note: The current system bottleneck is the UART link, resulting in the hardware being idle 99% of the time while waiting for data. The underlying architecture, however, proves that hardware-optimized math is significantly more efficient than general-purpose CPU compute.*

## Tech Stack
* **Hardware Description:** Verilog / RTL
* **Compute Platforms:** Altera/Intel Cyclone V FPGA
* **Software/Tooling:** Python (Inference Terminal), UART/Serial Communication, Int8 Quantization 
* **Architecture:** Systolic Arrays, Von Neumann Architecture Analysis, Transformer/Attention Mechanisms
  
## Architecture Overview

<p align="center">
  <sub><i>Scaled Dot-Product Attention Diagram</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/417b9bd9-37f4-4b59-807e-82215eea52f2" width="600" />
</p>

The system optimizes the **Scaled Dot-Product Attention** mechanism found in Transformer models, which requires a high volume of multiplications. By implementing hardware-level MatMul units, Q-Tensor accelerates the core mathematical requirements of modern LLMs.

<p align="center">
  <sub><i>High-Level Component Diagram</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/6817be4f-8d0b-4c37-b2e9-73fd62d8cf30" width="600" />
</p>

### FPGA Component Diagram
The internal RTL architecture consists of modules synchronized for deterministic I/O:
*  **UART Rx:** Handles the 115200 baud link for incoming data.
*  **Controller:** Orchestrates the data flow between the input driver and the compute core.
*  **Input Storage:** Acts as a buffer to feed the systolic array wavefront.
*  **Systolic Array Core:** The primary engine for parallelized matrix multiplication.
*  **Output Packer:** Collects processed data to be sent back via the UART output driver.
*  **UART Tx:** Handles the 115200 baud link for outgoing data.

## Software Integration

* **GPT-2 Injection Point**: Modified a pure NumPy GPT-2 pipeline to call the FPGA accelerator for matrix multiplication, enabling a working hybrid inference demo without heavyweight ML frameworks.
* **Quantized Data Path**: Inputs are quantized to int8 on the host, computed as int8 to int32 on FPGA, then converted back for integration. This mirrors real accelerator arithmetic: small inputs, wide accumulators.
* **Verification Harness**: Built a CPU reference path that runs the same quantized GEMM and compares element-by-element against FPGA output, ensuring bit-exact correctness across randomized tests.
  
---
*Developed by Jack Polloway and Faraz Fashizedeh partially for nwHacks 2026.*
