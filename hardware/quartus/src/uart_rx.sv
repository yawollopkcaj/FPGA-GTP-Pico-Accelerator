module uart_rx #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 115200
)(
    input  logic clk,
    input  logic rst,            // active-high reset
    input  logic rx,             // idle high
    output logic [7:0] data_out,
    output logic       data_valid
);

    // Rounded clocks per bit
    localparam int CLKS_PER_BIT = (CLK_HZ + (BAUD/2)) / BAUD;
    localparam int HALF_BIT     = CLKS_PER_BIT / 2;

    // Synchronize rx to clk (reduces metastability)
    logic rx_ff1, rx_ff2;
    always_ff @(posedge clk) begin
        rx_ff1 <= rx;
        rx_ff2 <= rx_ff1;
    end
    wire rx_s = rx_ff2;

    typedef enum logic [1:0] { IDLE, START, DATA, STOP } state_t;
    state_t state;

    int unsigned count;
    logic [2:0]  bit_idx;
    logic [7:0]  rx_buf;

    always_ff @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            count      <= 0;
            bit_idx    <= 0;
            rx_buf     <= 8'h00;
            data_out   <= 8'h00;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0; // pulse

            case (state)
                IDLE: begin
                    count   <= 0;
                    bit_idx <= 0;
                    if (rx_s == 1'b0) begin
                        state <= START;
                        count <= 0;
                    end
                end

                START: begin
                    // Sample in middle of start bit to confirm it's real
                    if (count == HALF_BIT) begin
                        if (rx_s == 1'b0) begin
                            state   <= DATA;
                            count   <= 0;
                            bit_idx <= 0;
                        end else begin
                            state <= IDLE; // glitch
                        end
                    end else begin
                        count <= count + 1;
                    end
                end

                DATA: begin
                    if (count == CLKS_PER_BIT-1) begin
                        count <= 0;

                        // LSB-first
                        rx_buf[bit_idx] <= rx_s;

                        if (bit_idx == 3'd7) begin
                            state <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        count <= count + 1;
                    end
                end

                STOP: begin
                    if (count == CLKS_PER_BIT-1) begin
                        count <= 0;

                        data_out   <= rx_buf;
                        data_valid <= 1'b1;
                        state      <= IDLE;
                    end else begin
                        count <= count + 1;
                    end
                end
            endcase
        end
    end

endmodule
