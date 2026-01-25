# Project Q-Tensor: Hardware Acceleration for LLM Inference

## Summary

**Project Q-Tensor** is a high-performance, energy-efficient hybrid inference engine designed to accelerate Large Language Model (LLM) operations using FPGA-based **systolic arrays**. By offloading computationally intensive General Matrix-Matrix Multiplication (GEMM) tasks from a host CPU to custom RTL, this project addresses the "Von Neumann Tax" (the performance bottleneck caused by constant data movement between the processor and main memory).

## The Problem We Set Out to Solve

LLM inference is dominated by matrix multiplication operations (GEMMs) within attention and linear layers. On general-purpose CPUs (Von Neumann architectures), this workload incurs significant control overhead, as substantial energy is spent on instruction fetching and decoding rather than arithmetic. Furthermore, the generative phase of LLMs is memory-bound due to low arithmetic intensity; the processor is frequently stalled waiting for weights to be fetched from memory, making DRAM bandwidth—rather than compute capacity—the primary bottleneck. 

<p align="center">
  <sub><i>Standard Memory Hiarchy Design</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f8b9e9ff-cad9-4119-8352-288286721726" width="400" />
</p>

*This issue has been addressed in large-scale data centers by Google's TPU (or NVIDIA's Tensor Cores), but the average developer is stuck using "unoptimized hardware"*

## Solution Highlights

<p align="center">
  <sub><i>Data Movement on the Systolic Array and FPGA Plane in Operation</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/53bc1dd6-7ad0-4c98-80dd-dd503108c81f" width="400" />
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
  <img src="https://github.com/user-attachments/assets/d3e0c325-b593-4fb4-9a86-a27e6249023c" width="400" />
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
  <img src="https://github.com/user-attachments/assets/472b30b5-ede4-4b56-82b5-5a0ad672b63b" width="400" />
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
  <img src="https://github.com/user-attachments/assets/417b9bd9-37f4-4b59-807e-82215eea52f2" width="200" />
</p>

The system optimizes the **Scaled Dot-Product Attention** mechanism found in Transformer models, which requires a high volume of multiplications. By implementing hardware-level MatMul units, Q-Tensor accelerates the core mathematical requirements of modern LLMs.

<p align="center">
  <sub><i>High-Level Component Diagram</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/695bb156-5592-43cd-aefa-f5eb9768133c" width="600" />
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

## Demo

To showcase the system end-to-end, we built a lightweight web interface that allows users to enter prompts and run live GPT-2 inference through the hybrid CPU–FPGA pipeline. The frontend acts as a control and visualization layer, sending prompts to the backend, displaying generated text in real time, and surfacing performance analytics such as token counts, latency, and FPGA accelerator utilization for each run.

<p align="center">
  <sub><i>nwHacks Demo: Promt-based GPT-2 Inference with Real-Time FPGA Performance Analytics</i></sub>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f9b8fa88-b1b9-4acd-a2e1-acb8e0d65992" width="600" />
</p>

## References

**[1]** N. P. Jouppi *et al.*, "In-datacenter performance analysis of a tensor processing unit," in *Proc. 44th Annu. Int. Symp. Comput. Archit. (ISCA)*, Toronto, ON, Canada, 2017, pp. 1–12. Available: https://arxiv.org/abs/1704.04760

**[2]** S. Markidis *et al.*, "NVIDIA Tensor Core programmability, performance & precision," in *Proc. IEEE Int. Parallel Distrib. Process. Symp. Workshops (IPDPSW)*, Vancouver, BC, Canada, 2018, pp. 522–531. Available: https://arxiv.org/abs/1803.04014

---
*Developed by Jack Polloway and Faraz Fashizedeh partially for nwHacks 2026.*
