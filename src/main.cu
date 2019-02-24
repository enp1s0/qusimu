#include <cooperative_groups.h>

// 命令は固定長
using inst_t = uint64_t;
using inst_type_t = uint64_t;
using qubit_t = float;
// unary命令
// |63        61|57    30|29       0|
// |  命令種別  | 未使用 | 計算対象 |
// binary命令
// |63        61|60        56|55    35|34          30|29       0|
// |  命令種別  | 同時発行数 | 未使用 | コントロール | 計算対象 |
// ternary命令
// |63        61|60        56|55    40|39          35|34          30|29       0|
// |  命令種別  | 同時発行数 | 未使用 | コントロール | コントロール | 計算対象 |

// 命令種別
constexpr inst_type_t inst_type_nil = 0x0;
constexpr inst_type_t inst_type_x   = 0x1;
constexpr inst_type_t inst_type_z   = 0x2;
constexpr inst_type_t inst_type_h   = 0x3;
constexpr inst_type_t inst_type_cx  = 0x4;
constexpr inst_type_t inst_type_cz  = 0x5;
constexpr inst_type_t inst_type_ccx = 0x6;

// スレッド tid が実行する命令を解読
__device__ inst_type_t decode_inst_type(const inst_t* const insts, std::size_t* const inst_index, const std::size_t tid){
	const auto mask = static_cast<inst_t>(1) << tid;
	const auto inst = insts[*inst_index];
	// |63   61|が命令種別なのでマジックナンバー61
	const auto inst_type = static_cast<inst_type_t>(inst >> 61);
	
	// unaryなら1回の呼び出しで関数終了
	if(inst_type < inst_type_cx){
		(*inst_index)++;
		return inst_type;
	}
	// binary, ternaryであれば同時発行数を考慮する
	const std::size_t num_parallel = (inst & (~(static_cast<inst_t>(1)<<61))) >> 56;
	(*inst_index) += num_parallel;
	for(std::size_t np = 0; np < num_parallel; np++){
		if(insts[(*inst_index) + np] && mask) return inst;
	}
	return inst_type_nil;
}

__device__ void convert_x(qubit_t* const qubits, const inst_t inst, const std::size_t tid){
	// 交換部分の解析
	constexpr auto mask = (~(static_cast<inst_t>(1)<<31));
	const auto xor_mask = inst & mask;

	// TODO : 書き込みと読み込みのどちらで結合アクセスを使うか
	// TODO : 実は処理が「交換」なので，並列数は半分で構わない
	qubits[tid ^ xor_mask] = qubits[xor_mask];
}
__device__ void convert_z(qubit_t* const qubits, const std::size_t k){
	qubits[k] = - qubits[k];
}

__global__ void qusimu_kernel(const inst_t* const insts, const std::size_t num_insts, const std::size_t N){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N){
		return;
	}
	// 全スレッドでgroupを作る
	const auto all_threads_group = cooperative_groups::coalesced_threads();
	// 命令実行ループ
	// プログラムカウンタ(inst_index)の加算処理はdecode_inst内で行う
	for(std::size_t inst_index = 0; inst_index < num_insts;){
		// デコード
		const auto inst_type = decode_inst_type(insts, &inst_index, tid);

		// 実行する命令がないならforに戻る
		if(inst_type == inst_type_nil) continue;

		// X
		if(inst_type == inst_type_x){
			continue;
		}

		// Z
		if(inst_type == inst_type_z){
			continue;
		}

		// H
		if(inst_type == inst_type_h){
			continue;
		}

		// CX
		if(inst_type == inst_type_cx){
			continue;
		}

		// CZ
		if(inst_type == inst_type_cz){
			continue;
		}

		// CCX
		if(inst_type == inst_type_ccx){
			continue;
		}
	}
}
