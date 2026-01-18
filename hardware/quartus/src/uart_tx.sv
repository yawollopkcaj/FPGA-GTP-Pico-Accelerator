module uart_tx #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 115200
)(
    input  logic clk,
    input  logic rst,         // active-high
    input  logic [7:0] data_in,
    input  logic       start, // 1-cycle pulse when idle
    output logic       tx,    // idle high
    output logic       busy,  // solid high during entire frame
    output logic       done   // 1-cycle pulse when frame completes
);

    localparam int CLKS_PER_BIT = (CLK_HZ + (BAUD/2)) / BAUD;

    typedef enum logic [1:0] { IDLE, START_BIT, DATA_BITS, STOP_BIT } state_t;
    state_t state;

    int unsigned count;
    logic [2:0]  bit_idx;
    logic [7:0]  shreg;

    always_ff @(posedge clk) begin
        if (rst) begin
            state   <= IDLE;
            tx      <= 1'b1;
            busy    <= 1'b0;
            done    <= 1'b0;
            count   <= 0;
            bit_idx <= 0;
            shreg   <= 8'h00;
        end else begin
            done <= 1'b0; // default: done is only high for 1 cycle

            case (state)
                IDLE: begin
                    tx   <= 1'b1;
                    busy <= 1'b0;
                    count <= 0;
                    bit_idx <= 0;
                    if (start) begin
                        shreg <= data_in;
                        busy  <= 1'b1;
                        tx    <= 1'b0; // start bit
                        state <= START_BIT;
                        count <= 0;
                    end
                end

                START_BIT: begin
                    busy <= 1'b1;
                    if (count == CLKS_PER_BIT-1) begin
                        count   <= 0;
                        bit_idx <= 0;
                        tx      <= shreg[0];
                        state   <= DATA_BITS;
                    end else begin
                        count <= count + 1;
                    end
                end

                DATA_BITS: begin
                    busy <= 1'b1;
                    if (count == CLKS_PER_BIT-1) begin
                        count <= 0;
                        if (bit_idx == 3'd7) begin
                            tx    <= 1'b1;   // stop bit
                            state <= STOP_BIT;
                        end else begin
                            bit_idx <= bit_idx + 1;
                            tx      <= shreg[bit_idx + 1];
                        end
                    end else begin
                        count <= count + 1;
                    end
                end

                STOP_BIT: begin
                    busy <= 1'b1;
                    if (count == CLKS_PER_BIT-1) begin
                        count <= 0;
                        tx    <= 1'b1;
                        busy  <= 1'b0;
                        done  <= 1'b1;  // pulse done for 1 cycle
                        state <= IDLE;
                    end else begin
                        count <= count + 1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule