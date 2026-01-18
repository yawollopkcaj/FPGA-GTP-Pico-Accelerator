import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "fpga_timing.csv"
MS = 1000.0

tokens = []
total_ms = []
no_uart_ms = []
uart_ms = []

with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_total = float(row["total_s"]) * MS
        t_uart = float(row.get("tile_uart_wall_s", 0.0)) * MS

        tokens.append(int(row["token_id"]))
        total_ms.append(t_total)
        uart_ms.append(t_uart)
        no_uart_ms.append(max(0.0, t_total - t_uart))

tokens = np.arange(1, len(tokens) + 1)
total_ms = np.array(total_ms)
no_uart_ms = np.array(no_uart_ms)
uart_ms = np.array(uart_ms)

# ----------------------------
# Plot: UART impact per token
# ----------------------------
plt.figure(figsize=(10, 4))

width = 0.38
plt.bar(tokens - width/2, total_ms, width, label="Measured token total (ms)")
plt.bar(tokens + width/2, no_uart_ms, width, label="Estimated w/o UART (ms)")

plt.xlabel("Token #")
plt.ylabel("Time (ms)")
plt.title("UART impact per token (measured vs estimated without UART)")
plt.xticks(tokens)
plt.legend()
plt.tight_layout()
plt.savefig("uart_impact_per_token.png", dpi=200)
plt.show()

# ----------------------------
# Optional: UART-only view
# ----------------------------
plt.figure(figsize=(10, 3))
plt.bar(tokens, uart_ms)
plt.xlabel("Token #")
plt.ylabel("UART wall time (ms)")
plt.title("UART wall-time per token (startup penalty visible on token 1)")
plt.tight_layout()
plt.savefig("uart_wall_per_token.png", dpi=200)
plt.show()
