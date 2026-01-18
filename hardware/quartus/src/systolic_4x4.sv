module systolic_4x4 (
    input  logic clk,
    input  logic rst,

    // Inject 4 A values (one per row) and 4 B values (one per col) each cycle
    input  logic signed [7:0] a_in_row [0:3],
    input  logic signed [7:0] b_in_col [0:3],

    // 16 outputs (row-major): c_out[0]=C00 ... c_out[15]=C33
    output logic signed [31:0] c_out [0:15]
);

    // Interconnect wires between PEs
    logic signed [7:0]  a_bus [0:3][0:4]; // [row][col tap], 0 is input, 4 is after last PE
    logic signed [7:0]  b_bus [0:4][0:3]; // [row tap][col], 0 is input, 4 is after last PE
    logic signed [31:0] sum   [0:3][0:3];

    // Connect external inputs to left/top edges
    genvar r, c;

    generate
        for (r = 0; r < 4; r++) begin : EDGE_A
            always_comb a_bus[r][0] = a_in_row[r];
        end
        for (c = 0; c < 4; c++) begin : EDGE_B
            always_comb b_bus[0][c] = b_in_col[c];
        end
    endgenerate

    // Instantiate 16 PEs
    generate
        for (r = 0; r < 4; r++) begin : ROW
            for (c = 0; c < 4; c++) begin : COL
                pe u_pe (
                    .clk   (clk),
                    .rst   (rst),
                    .a_in  (a_bus[r][c]),
                    .b_in  (b_bus[r][c]),
                    .a_out (a_bus[r][c+1]),
                    .b_out (b_bus[r+1][c]),
                    .sum   (sum[r][c])
                );
            end
        end
    endgenerate

    // Map sums to row-major output array
    always_comb begin
        c_out[0]  = sum[0][0];
        c_out[1]  = sum[0][1];
        c_out[2]  = sum[0][2];
        c_out[3]  = sum[0][3];

        c_out[4]  = sum[1][0];
        c_out[5]  = sum[1][1];
        c_out[6]  = sum[1][2];
        c_out[7]  = sum[1][3];

        c_out[8]  = sum[2][0];
        c_out[9]  = sum[2][1];
        c_out[10] = sum[2][2];
        c_out[11] = sum[2][3];

        c_out[12] = sum[3][0];
        c_out[13] = sum[3][1];
        c_out[14] = sum[3][2];
        c_out[15] = sum[3][3];
    end

endmodule
