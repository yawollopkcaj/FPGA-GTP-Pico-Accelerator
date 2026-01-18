module pe (
    input  logic clk,
    input  logic rst,
    input  logic signed [7:0]  a_in,
    input  logic signed [7:0]  b_in,
    output logic signed [7:0]  a_out,
    output logic signed [7:0]  b_out,
    output logic signed [31:0] sum
);
    always_ff @(posedge clk) begin
        if (rst) begin
            sum   <= 32'sd0;
            a_out <= 8'sd0;
            b_out <= 8'sd0;
        end else begin
            sum   <= sum + (a_in * b_in);
            a_out <= a_in;
            b_out <= b_in;
        end
    end
endmodule
