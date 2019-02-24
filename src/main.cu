#include <iostream>
#include <cooperative_groups.h>
// CUDAの組み込み関数はconstexprではないので
constexpr float sqrt2 = 1.41421356237f;

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

__device__ void convert_x(qubit_t* const qubits, const inst_t inst, const std::size_t tid, const cooperative_groups::coalesced_group &all_threads_group){
	// 交換部分の解析
	constexpr auto mask = (~(static_cast<inst_t>(1)<<31));
	const auto xor_mask = inst & mask;

	// TODO : 書き込みと読み込みのどちらで結合アクセスを使うか
	// TODO : 実は処理が「交換」なので，並列数は半分で構わない
	const auto tmp = qubits[tid];
	all_threads_group.sync();
	qubits[tid ^ xor_mask] = tmp;
}
__device__ void convert_z(qubit_t* const qubits, const inst_t inst, const std::size_t tid){
	constexpr auto mask = (~(static_cast<inst_t>(1)<<31));
	const auto target_bits = inst & mask;

	if((tid & target_bits) != 0){
		// TODO : 先頭ビット反転とどちらが速いか
		qubits[tid] = -qubits[tid];
	}
}
__device__ void convert_h(qubit_t* const qubits, const inst_t inst, const std::size_t tid, const cooperative_groups::coalesced_group &all_threads_group){
	// 交換部分の解析
	constexpr auto mask = (~(static_cast<inst_t>(1)<<31));
	const auto target_bits = inst & mask;

	// TODO : 書き込みと読み込みのどちらで結合アクセスを使うか
	// TODO : 実は処理が「交換」なので，並列数は半分で構わない
	const auto p0 = qubits[tid];
	const auto p1 = qubits[tid ^ target_bits];
	all_threads_group.sync();
	if((tid & target_bits) != 0){
		qubits[tid] = (p0 + p1) / sqrt2;
	}else{
		qubits[tid] = (p0 - p1) / sqrt2;
	}
}
__device__ void convert_cx(qubit_t* const qubits, const inst_t inst, const std::size_t tid, const cooperative_groups::coalesced_group &all_threads_group){
	constexpr auto mask = (~(static_cast<inst_t>(1)<<31));
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits = static_cast<inst_t>(1) << ((inst >> 30) & 0x1f);

	if(tid & ctrl_bits == 0){
		return;
	}
	const auto p = qubits[tid ^ target_bits];
	all_threads_group.sync();
	qubits[tid] = p;
}
__device__ void convert_cz(qubit_t* const qubits, const inst_t inst, const std::size_t tid){
	constexpr auto mask = (~(static_cast<inst_t>(1)<<31));
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits = static_cast<inst_t>(1) << ((inst >> 30) & 0x1f);

	if(tid & ctrl_bits == 0){
		return;
	}
	qubits[tid] = -qubits[tid];
}

__global__ void qusimu_kernel(qubit_t* const qubits, const inst_t* const insts, const std::size_t num_insts, const std::size_t N){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N){
		return;
	}
	// 全スレッドでgroupを作る
	const auto all_threads_group = cooperative_groups::coalesced_threads();
	// 命令実行ループ
	// プログラムカウンタ(inst_index)の加算処理はdecode_inst内で行う
	for(std::size_t inst_index = 0; inst_index < num_insts;){
		all_threads_group.sync();
		// デコード
		const auto inst = insts[inst_index];
		// |63   61|が命令種別なのでマジックナンバー61
		const auto inst_type = static_cast<inst_type_t>(inst >> 61);

		// X
		if(inst_type == inst_type_x){
			convert_x(qubits, inst, tid, all_threads_group);
			continue;
		}

		// Z
		if(inst_type == inst_type_z){
			convert_z(qubits, inst, tid);
			continue;
		}

		// H
		if(inst_type == inst_type_h){
			convert_h(qubits, inst, tid, all_threads_group);
			continue;
		}

		// CX
		if(inst_type == inst_type_cx){
			convert_cx(qubits, inst, tid, all_threads_group);
			continue;
		}

		// CZ
		if(inst_type == inst_type_cz){
			convert_cz(qubits, inst, tid);
			continue;
		}

		// CCX
		if(inst_type == inst_type_ccx){
			continue;
		}
	}
}

int main(){}
