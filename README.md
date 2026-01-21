# Project Q-Tensor: Hardware Acceleration for LLM Inference

[cite_start]**Project Q-Tensor** is a high-performance, energy-efficient hybrid inference engine designed to accelerate Large Language Model (LLM) operations using FPGA-based systolic arrays[cite: 1, 2, 79]. [cite_start]By offloading computationally intensive General Matrix-Matrix Multiplication (GEMM) tasks from a host CPU to custom RTL, this project addresses the "Von Neumann Tax"—the performance bottleneck caused by constant data movement between the processor and main memory[cite: 8, 78, 117].

## 🚀 Technical Highlights
* [cite_start]**Custom Systolic Array:** Developed a scalable systolic array architecture in Verilog to maximize data reuse and achieve 100% compute utilization during matrix operations[cite: 79, 110, 112].
* [cite_start]**Hybrid Inference Engine:** Engineered a system that allows for a hybrid backend using Int8 quantization to offload MAC operations[cite: 206, 210, 224].
* [cite_start]**Energy Efficiency:** Achieved a **2.2X reduction in energy consumption** per $4 \times 4$ GEMM tile compared to traditional CPU execution[cite: 128, 129, 132].
* [cite_start]**Performance:** Demonstrated a hardware compute-only speedup reaching an effective throughput of **0.320 GMAC/s**[cite: 239].
* [cite_start]**Power Savings:** Realized a saving of approximately **65 mW per $4 \times 4$ tile** compared to CPU-based compute[cite: 134].

## 🏗️ Architecture Overview
[cite_start]The system optimizes the **Scaled Dot-Product Attention** mechanism found in Transformer models, which requires a high volume of multiplications[cite: 51, 52, 53]. [cite_start]By implementing hardware-level **MatMul** units, Q-Tensor accelerates the core mathematical requirements of modern LLMs[cite: 54, 58, 212].

### FPGA Component Diagram
[cite_start]The internal RTL architecture consists of modules synchronized for deterministic I/O[cite: 80, 99]:
1.  [cite_start]**UART Input Driver:** Handles the 115200 baud link for incoming data[cite: 100, 227].
2.  [cite_start]**Controller:** Orchestrates the data flow between the input driver and the compute core[cite: 101].
3.  [cite_start]**Input Storage:** Acts as a buffer to feed the systolic array wavefront[cite: 102].
4.  [cite_start]**Systolic Array Core:** The primary engine for parallelized matrix multiplication[cite: 107].
5.  [cite_start]**Output Packer:** Collects processed data to be sent back via the UART output driver[cite: 105, 106].

## 📊 Performance Benchmarks
| Metric | CPU (Fallback) | FPGA Accelerator (Hybrid) |
| :--- | :--- | :--- |
| **Energy per $4 \times 4$ GEMM Tile** | [cite_start]$\sim 26\ nJ$ [cite: 115, 128] | [cite_start]$\sim 12\ nJ$ [cite: 115, 128] |
| **Compute Time (Micro-benchmark)** | [cite_start]$2.33\ \mu s$ (Int8) [cite: 153, 163] | [cite_start]$\sim 0.20\ \mu s$ (Est.) [cite: 153, 164] |
| **Utilization** | [cite_start]Sequential / Cache-dependent [cite: 13] | [cite_start]100% Utilization [cite: 112] |

[cite_start]*Note: The current system bottleneck is the UART link, resulting in the hardware being idle 99% of the time while waiting for data[cite: 165, 171, 172]. [cite_start]The underlying architecture, however, proves that hardware-optimized math is significantly more efficient than general-purpose CPU compute[cite: 175, 177].*

## 🗺️ Roadmap: Toward "Net Zero Autonomy"
[cite_start]The roadmap to evolve Q-Tensor into a production-grade "AI Box" includes[cite: 179, 187, 192]:
* [cite_start]**Phase 1:** Replacing the UART connection with high-bandwidth **PCIe or DMA** to eliminate I/O bottlenecks[cite: 180, 181, 182].
* [cite_start]**Phase 2:** Transitioning to fully FPGA-based calculations for large-scale $1M \times 1M$ GEMM operations[cite: 183, 184, 185].
* [cite_start]**Phase 3:** Developing a standalone, fully isolated system using **Nios II Soft-core CPUs** and on-board **DDR4 RAM** to host complete models (e.g., GPT-2)[cite: 186, 188, 198, 200].

## 🛠️ Tech Stack
* [cite_start]**Hardware Description:** Verilog / RTL [cite: 99]
* [cite_start]**Compute Platforms:** Altera/Intel Cyclone V FPGA [cite: 85]
* [cite_start]**Software/Tooling:** Python (Inference Terminal), UART/Serial Communication, Int8 Quantization [cite: 206, 210, 214, 227]
* [cite_start]**Architecture:** Systolic Arrays, Von Neumann Architecture Analysis, Transformer/Attention Mechanisms [cite: 8, 53, 79]

---
[cite_start]*Developed by Jack Polloway and Faraz[cite: 3].*
