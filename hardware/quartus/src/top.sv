module top (
    input  logic CLOCK_50,

    input  logic uart_rx,   // PIN_F14
    output logic uart_tx,   // PIN_F15

    output logic [9:0] LEDR
);

    // -------------------------
    // Reset
    // -------------------------
    logic rst;
    assign rst = 1'b0;

    // -------------------------
    // UART RX
    // -------------------------
    logic [7:0] rx_data;
    logic       rx_valid;

    uart_rx #(
        .CLK_HZ(50_000_000),
        .BAUD  (115200)
    ) u_rx (
        .clk        (CLOCK_50),
        .rst        (rst),
        .rx         (uart_rx),
        .data_out   (rx_data),
        .data_valid (rx_valid)
    );

    // -------------------------
    // UART TX
    // -------------------------
    logic [7:0] tx_data;
    logic       tx_start;
    logic       tx_busy;
    logic       tx_done;

    uart_tx #(
        .CLK_HZ(50_000_000),
        .BAUD  (115200)
    ) u_tx (
        .clk     (CLOCK_50),
        .rst     (rst),
        .data_in (tx_data),
        .start   (tx_start),
        .tx      (uart_tx),
        .busy    (tx_busy),
        .done    (tx_done)
    );

    // -------------------------
    // Storage for A and B (int8)
    // -------------------------
    logic signed [7:0] A [0:3][0:3];
    logic signed [7:0] B [0:3][0:3];

    // -------------------------
    // Systolic array
    // -------------------------
    logic compute_rst;
    logic signed [7:0]  a_in_row [0:3];
    logic signed [7:0]  b_in_col [0:3];
    logic signed [31:0] c_out    [0:15];

    systolic_4x4 u_sys (
        .clk      (CLOCK_50),
        .rst      (compute_rst),
        .a_in_row (a_in_row),
        .b_in_col (b_in_col),
        .c_out    (c_out)
    );

    // -------------------------
    // Cycle counter for performance measurement
    // -------------------------
    logic [31:0] compute_cycles;

    // -------------------------
    // FSM States
    // -------------------------
    typedef enum logic [3:0] {
        WAIT_A5,
        RECV_PAYLOAD,
        RECV_DONE,       // Wait one cycle for last store to complete
        CLEAR_SYS,       // Assert reset
        WAIT_CLEAR,      // Wait one cycle for reset to complete
        RUN_SYS,
        SEND_ACK,
        WAIT_ACK_DONE,
        SEND_DATA,
        WAIT_DATA_DONE,
        SEND_CYCLES,
        WAIT_CYCLES_DONE,
        SEND_END,
        WAIT_END_DONE
    } state_t;

    state_t state;

    logic [5:0] rx_idx;    // 0..31 for receiving payload
    logic [3:0] t_cycle;   // 0..9 for systolic timing
    logic [6:0] tx_idx;    // 0..63 for data bytes
    logic [1:0] cyc_idx;   // 0..3 for cycle count bytes

    // Debug counters
    logic [7:0] dbg_tx_count;
    logic       dbg_end_sent;

    // -------------------------
    // Helper: get result byte (little-endian)
    // -------------------------
    function automatic logic [7:0] get_c_byte(input logic [6:0] byte_index);
        logic [3:0]  word_i;
        logic [1:0]  byte_i;
        logic [31:0] w;
        begin
            word_i = byte_index[6:2];  // which int32 (0-15)
            byte_i = byte_index[1:0];  // which byte within int32 (0-3)
            w = c_out[word_i];
            case (byte_i)
                2'd0: get_c_byte = w[7:0];
                2'd1: get_c_byte = w[15:8];
                2'd2: get_c_byte = w[23:16];
                default: get_c_byte = w[31:24];
            endcase
        end
    endfunction

    // -------------------------
    // Helper: get cycle count byte (little-endian)
    // -------------------------
    function automatic logic [7:0] get_cycle_byte(input logic [1:0] byte_index);
        case (byte_index)
            2'd0: get_cycle_byte = compute_cycles[7:0];
            2'd1: get_cycle_byte = compute_cycles[15:8];
            2'd2: get_cycle_byte = compute_cycles[23:16];
            default: get_cycle_byte = compute_cycles[31:24];
        endcase
    endfunction

    // -------------------------
    // Systolic input scheduling (skewed feed for A and B)
    // For C = A @ B:
    //   - a_in_row[i] feeds row i of A horizontally
    //   - b_in_col[j] feeds column j of B vertically
    // Skewing: at time t, row i gets A[i][t-i], col j gets B[t-j][j]
    // -------------------------
    integer i;
    always_comb begin
        for (i = 0; i < 4; i++) begin
            a_in_row[i] = 8'sd0;
            b_in_col[i] = 8'sd0;
        end

        if (state == RUN_SYS) begin
            for (i = 0; i < 4; i++) begin
                if ((t_cycle >= i) && ((t_cycle - i) < 4))
                    a_in_row[i] = A[i][t_cycle - i];
            end
            for (i = 0; i < 4; i++) begin
                if ((t_cycle >= i) && ((t_cycle - i) < 4))
                    b_in_col[i] = B[t_cycle - i][i];
            end
        end
    end

    // -------------------------
    // Main FSM
    // -------------------------
    always_ff @(posedge CLOCK_50) begin
        // Default: tx_start is only high for 1 cycle
        tx_start    <= 1'b0;
        compute_rst <= 1'b0;

        if (rst) begin
            state          <= WAIT_A5;
            rx_idx         <= 6'd0;
            t_cycle        <= 4'd0;
            tx_idx         <= 7'd0;
            cyc_idx        <= 2'd0;
            tx_data        <= 8'h00;
            compute_cycles <= 32'd0;
            dbg_tx_count   <= 8'd0;
            dbg_end_sent   <= 1'b0;
        end else begin
            case (state)

                // -------------------------
                // Wait for start byte 0xA5
                // -------------------------
                WAIT_A5: begin
                    rx_idx       <= 6'd0;
                    dbg_tx_count <= 8'd0;
                    dbg_end_sent <= 1'b0;
                    if (rx_valid && rx_data == 8'hA5) begin
                        state <= RECV_PAYLOAD;
                    end
                end

                // -------------------------
                // Receive 32 bytes: A[0..15], B[16..31]
                // Both stored row-major: index/4 = row, index%4 = col
                // -------------------------
                RECV_PAYLOAD: begin
                    if (rx_valid) begin
                        if (rx_idx < 6'd16) begin
                            // A matrix: bytes 0-15 -> A[0][0] to A[3][3]
                            A[rx_idx[3:2]][rx_idx[1:0]] <= $signed(rx_data);
                        end else begin
                            // B matrix: bytes 16-31 -> B[0][0] to B[3][3]
                            // rx_idx=16 -> B[0][0], rx_idx=17 -> B[0][1], etc.
                            B[rx_idx[3:2]][rx_idx[1:0]] <= $signed(rx_data);
                        end

                        if (rx_idx == 6'd31) begin
                            state <= RECV_DONE;  // Go to wait state
                        end else begin
                            rx_idx <= rx_idx + 6'd1;
                        end
                    end
                end

                // -------------------------
                // Wait one cycle for last store to complete
                // -------------------------
                RECV_DONE: begin
                    state <= CLEAR_SYS;
                end

                // -------------------------
                // Assert reset to clear systolic array accumulators
                // -------------------------
                CLEAR_SYS: begin
                    compute_rst    <= 1'b1;
                    t_cycle        <= 4'd0;
                    compute_cycles <= 32'd0;
                    state          <= WAIT_CLEAR;
                end

                // -------------------------
                // Wait for reset to complete (compute_rst goes low)
                // -------------------------
                WAIT_CLEAR: begin
                    // compute_rst defaults to 0, so it's low this cycle
                    // Now safe to start feeding
                    state <= RUN_SYS;
                end

                // -------------------------
                // Run systolic array (10 cycles for 4x4)
                // -------------------------
                RUN_SYS: begin
                    compute_cycles <= compute_cycles + 32'd1;
                    if (t_cycle == 4'd9) begin
                        state <= SEND_ACK;
                    end else begin
                        t_cycle <= t_cycle + 4'd1;
                    end
                end

                // -------------------------
                // Send ACK (0x5A)
                // -------------------------
                SEND_ACK: begin
                    if (!tx_busy) begin
                        tx_data      <= 8'h5A;
                        tx_start     <= 1'b1;
                        tx_idx       <= 7'd0;
                        dbg_tx_count <= dbg_tx_count + 8'd1;
                        state        <= WAIT_ACK_DONE;
                    end
                end

                WAIT_ACK_DONE: begin
                    if (tx_done) begin
                        state <= SEND_DATA;
                    end
                end

                // -------------------------
                // Send 64 data bytes (16 x int32)
                // -------------------------
                SEND_DATA: begin
                    tx_data      <= get_c_byte(tx_idx);
                    tx_start     <= 1'b1;
                    dbg_tx_count <= dbg_tx_count + 8'd1;
                    state        <= WAIT_DATA_DONE;
                end

                WAIT_DATA_DONE: begin
                    if (tx_done) begin
                        if (tx_idx == 7'd63) begin
                            cyc_idx <= 2'd0;
                            state   <= SEND_CYCLES;
                        end else begin
                            tx_idx <= tx_idx + 7'd1;
                            state  <= SEND_DATA;
                        end
                    end
                end

                // -------------------------
                // Send 4 cycle count bytes
                // -------------------------
                SEND_CYCLES: begin
                    tx_data      <= get_cycle_byte(cyc_idx);
                    tx_start     <= 1'b1;
                    dbg_tx_count <= dbg_tx_count + 8'd1;
                    state        <= WAIT_CYCLES_DONE;
                end

                WAIT_CYCLES_DONE: begin
                    if (tx_done) begin
                        if (cyc_idx == 2'd3) begin
                            state <= SEND_END;
                        end else begin
                            cyc_idx <= cyc_idx + 2'd1;
                            state   <= SEND_CYCLES;
                        end
                    end
                end

                // -------------------------
                // Send END marker (0xA6)
                // -------------------------
                SEND_END: begin
                    tx_data      <= 8'hA6;
                    tx_start     <= 1'b1;
                    dbg_tx_count <= dbg_tx_count + 8'd1;
                    dbg_end_sent <= 1'b1;
                    state        <= WAIT_END_DONE;
                end

                WAIT_END_DONE: begin
                    if (tx_done) begin
                        state <= WAIT_A5;
                    end
                end

                default: state <= WAIT_A5;
            endcase
        end
    end

    // -------------------------
    // Debug LEDs (active HIGH on DE0-CV)
    // -------------------------
    always_comb begin
        LEDR = 10'b0;

        LEDR[0] = (state == RECV_PAYLOAD);
        LEDR[1] = (state == RUN_SYS);
        LEDR[2] = (state == SEND_ACK)    || (state == WAIT_ACK_DONE) ||
                  (state == SEND_DATA)   || (state == WAIT_DATA_DONE) ||
                  (state == SEND_CYCLES) || (state == WAIT_CYCLES_DONE) ||
                  (state == SEND_END)    || (state == WAIT_END_DONE);

        LEDR[7:3] = dbg_tx_count[4:0];
        LEDR[8]   = dbg_end_sent;
        LEDR[9]   = tx_busy;
    end

endmodule