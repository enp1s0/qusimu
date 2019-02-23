
// 命令は固定長
using instruction_t = uint64_t;
// unary命令
// |63        58|57    30|29       0|
// |  命令種別  | 未使用 | 計算対象 |
// binary命令
// |63        58|57    35|34          30|29       0|
// |  命令種別  | 未使用 | コントロール | 計算対象 |
// binary命令
// |63        58|57    35|34          30|29       0|
// |  命令種別  | 未使用 | コントロール | 計算対象 |
// ternary命令
// |63        58|57    40|39          35|34          30|29       0|
// |  命令種別  | 未使用 | コントロール | コントロール | 計算対象 |

// 命令種別
constexpr instruction_t inst_x   = 0b000001;
constexpr instruction_t inst_z   = 0b000010;
constexpr instruction_t inst_h   = 0b000100;
constexpr instruction_t inst_cx  = 0b001000;
constexpr instruction_t inst_cz  = 0b010000;
constexpr instruction_t inst_ccx = 0b100000;

__global__ void qusimu_kernel(const instruction_t* const instructions, const std::size_t num_instructions){

}
