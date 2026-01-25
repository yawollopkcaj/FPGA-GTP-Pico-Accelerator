# Project Q-Tensor: Hardware Acceleration for LLM Inference

## Summary

**Project Q-Tensor** is a high-performance, energy-efficient hybrid inference engine designed to accelerate Large Language Model (LLM) operations using FPGA-based **systolic arrays**. By offloading computationally intensive General Matrix-Matrix Multiplication (GEMM) tasks from a host CPU to custom RTL, this project addresses the "Von Neumann Tax" (the performance bottleneck caused by constant data movement between the processor and main memory).

## The Problem

LLM inference is dominated by matrix multiplication (attention and linear layers). On conventional CPUs, each multiply-accumulate step requires fetching instructions and operands from memory, creating significant overhead that limits throughput and wastes energy. The CPU is idle (waiting for memory transfer) more than actively computing.

## The Solution

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

*Note: The current system bottleneck is the UART link, resulting in the hardware being idle 99% of the time while waiting for data. The underlying architecture, however, proves that hardware-optimized math is significantly more efficient than general-purpose CPU compute.*

## Architecture Overview

The system optimizes the **Scaled Dot-Product Attention** mechanism found in Transformer models, which requires a high volume of multiplications. By implementing hardware-level **MatMul** units, Q-Tensor accelerates the core mathematical requirements of modern LLMs.

### FPGA Component Diagram
The internal RTL architecture consists of modules synchronized for deterministic I/O[:
1.  **UART Input Driver:** Handles the 115200 baud link for incoming data.
2.  **Controller:** Orchestrates the data flow between the input driver and the compute core.
3.  **Input Storage:** Acts as a buffer to feed the systolic array wavefront.
4.  **Systolic Array Core:** The primary engine for parallelized matrix multiplication.
5.  **Output Packer:** Collects processed data to be sent back via the UART output driver.


## Roadmap: Toward "Net Zero Autonomy"
The roadmap to evolve Q-Tensor into a production-grade LLM accelerator includes:
* **Phase 1:** Replacing the UART connection with high-bandwidth **PCIe or DMA** to eliminate I/O bottlenecks.
* **Phase 2:** Transitioning to fully FPGA-based calculations for large-scale $1M \times 1M$ GEMM operations.

## Tech Stack
* **Hardware Description:** Verilog / RTL
* **Compute Platforms:** Altera/Intel Cyclone V FPGA
* **Software/Tooling:** Python (Inference Terminal), UART/Serial Communication, Int8 Quantization 
* **Architecture:** Systolic Arrays, Von Neumann Architecture Analysis, Transformer/Attention Mechanisms

---
*Developed by Jack Polloway and Faraz Fashizedeh for nwHacks 2026.*
