#include <iostream>
#include <string>
#include <vector>
#include <cooperative_groups.h>
#include <cutf/memory.hpp>

#define DEBUG

// CUDAの組み込み関数はconstexprではないので
constexpr float sqrt2 = 1.41421356237f;
// スレッド数
// ATSUKANを走らせてもいいかも?
constexpr std::size_t num_threads_per_block = 256;

// 命令は固定長
using inst_t = uint64_t;
using inst_type_t = uint64_t;
using qubit_t = float;
// unary命令
// |63        61|57    32|31       0|
// |  命令種別  | 未使用 | 計算対象 |
// binary命令
// |63        61|60    37|36          32|31       0|
// |  命令種別  | 未使用 | コントロール | 計算対象 |
// ternary命令
// |63        61|60    43|41          37|36          32|31       0|
// |  命令種別  | 未使用 | コントロール | コントロール | 計算対象 |

// 命令種別
constexpr inst_type_t inst_type_x   = 0x1;
constexpr inst_type_t inst_type_z   = 0x2;
constexpr inst_type_t inst_type_h   = 0x3;
constexpr inst_type_t inst_type_cx  = 0x4;
constexpr inst_type_t inst_type_cz  = 0x5;
constexpr inst_type_t inst_type_ccx = 0x6;

// デバッグ用
__host__ __device__ void debug_print_inst(const inst_t inst){
	auto log2 = [] __host__ __device__ (const inst_t i){std::size_t l = 0;for(inst_t t = 1; !(i & t); t <<= 1, l++);return l;};
	printf("/*0x%lx*/ ", inst);
	const auto inst_type = inst >> 61;
	if(inst_type == inst_type_x) printf("X %lu", log2(inst & 0x3fffffff));
	if(inst_type == inst_type_z) printf("Z %lu", log2(inst & 0x3fffffff));
	if(inst_type == inst_type_h) printf("H %lu", log2(inst & 0x3fffffff));
	if(inst_type == inst_type_cx) printf("CX %lu %lu", ((inst >> 32) & 0x1f), log2(inst & 0x3fffffff));
	if(inst_type == inst_type_cz) printf("CZ %lu %lu", ((inst >> 32) & 0x1f), log2(inst & 0x3fffffff));
	if(inst_type == inst_type_ccx) printf("CCX %lu %lu %lu", ((inst >> 32) & 0x1f), ((inst >> 37) & 0x1f), log2(inst & 0x3fffffff));
	printf("\n");
}
void debug_print_insts(const std::vector<inst_t>& insts){
	printf("loaded instructions\n");
	printf("line /*  hex inst code   */ | decoded code |\n");
	printf("--------------------------------------------\n");
	std::size_t i = 0;
	for(const auto inst : insts){
		printf("%4lu ", (i++));
		debug_print_inst(inst);
	}
}

// Provided
__device__ float warpReduceMax(float val){
	for (int offset = warpSize/2; offset > 0; offset /= 2)
#if __CUDACC_VER_MAJOR__ >= 9
		val = fmaxf(val, __shfl_down_sync(~0, val, offset));
#else
	val = fmaxf(val, __shfl_down(val, offset));
#endif
	return val;
}

__global__ void maxabs(float *A, float *m){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int lane = threadIdx.x % warpSize;
	float val = fabsf(A[i]);
	val = warpReduceMax(val);
	if(lane == 0) atomicMax((int *) m, *(int *) &val);
}

__device__ void convert_x(qubit_t* const qubits, const inst_t inst, const std::size_t tid, const cooperative_groups::grid_group &all_threads_group){
	// 交換部分の解析
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_mask = inst & mask;

	// TODO : 書き込みと読み込みのどちらで結合アクセスを使うか
	// TODO : 実は処理が「交換」なので，並列数は半分で構わない
	const auto tmp = qubits[tid];
	all_threads_group.sync();
	qubits[tid ^ target_mask] = tmp;
}
__device__ void convert_z(qubit_t* const qubits, const inst_t inst, const std::size_t tid){
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;

	if((tid & target_bits) != 0){
		// TODO : 先頭ビット反転とどちらが速いか
		qubits[tid] = -qubits[tid];
	}
}
__device__ void convert_h(qubit_t* const qubits, const inst_t inst, const std::size_t tid, const cooperative_groups::grid_group &all_threads_group){
	// 交換部分の解析
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;

	// TODO : 書き込みと読み込みのどちらで結合アクセスを使うか
	// TODO : 実は処理が「交換」なので，並列数は半分で構わない
	const auto p0 = qubits[tid ^ target_bits];
	const auto p1 = qubits[tid];
	all_threads_group.sync();
	if((tid & target_bits) == 0){
		qubits[tid] = (p0 + p1) / sqrt2;
	}else{
		qubits[tid] = (p0 - p1) / sqrt2;
	}
}
__device__ void convert_cx(qubit_t* const qubits, const inst_t inst, const std::size_t tid, const cooperative_groups::grid_group &all_threads_group){
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits = static_cast<inst_t>(1) << ((inst >> 32) & 0x1f);

	if(tid & ctrl_bits == 0){
		return;
	}
	const auto p = qubits[tid ^ target_bits];
	all_threads_group.sync();
	qubits[tid] = p;
}
__device__ void convert_cz(qubit_t* const qubits, const inst_t inst, const std::size_t tid){
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits = static_cast<inst_t>(1) << ((inst >> 32) & 0x1f);

	if(tid & ctrl_bits == 0 || tid & target_bits == 0){
		return;
	}
	qubits[tid] = -qubits[tid];
}
__device__ void convert_ccx(qubit_t* const qubits, const inst_t inst, const std::size_t tid, const cooperative_groups::grid_group &all_threads_group){
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits_0 = static_cast<inst_t>(1) << ((inst >> 32) & 0x1f);
	const auto ctrl_bits_1 = static_cast<inst_t>(1) << ((inst >> 37) & 0x1f);

	if(tid & ctrl_bits_0 == 0 || tid & ctrl_bits_1 == 0){
		return;
	}
	const auto p = qubits[tid ^ target_bits];
	all_threads_group.sync();
	qubits[tid] = p;
}

__global__ void qusimu_kernel(qubit_t* const qubits, const inst_t* const insts, const std::size_t num_insts, const std::size_t N){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N){
		return;
	}
	// 初期化
	qubits[tid] = static_cast<qubit_t>(0);
	if(tid == 0){
		qubits[tid] = static_cast<qubit_t>(1);
	}
	// 全スレッドでgroupを作る
	const auto all_threads_group = cooperative_groups::this_grid();
	// 命令実行ループ
	for(std::size_t inst_index = 0; inst_index < num_insts; inst_index++){
		all_threads_group.sync();
		// デコード
		// 全スレッドが同じアドレスへアクセスするためキャッシュをうまく使いましょう
		const auto inst = __ldg(insts + inst_index);
		// |63   61|が命令種別なのでマジックナンバー61
		const auto inst_type = static_cast<inst_type_t>(inst >> 61);

#ifdef DEBUG
		if(tid == 0)
			debug_print_inst(inst);
#endif

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
			convert_ccx(qubits, inst, tid, all_threads_group);
			continue;
		}
	}
}

int main(){
	std::size_t n, k;
	std::cin >> n >> k;

	// 量子ビットの組み合わせ総数
	const std::size_t N = 1 << n;

	// 量子ビット on デバイスメモリ
	auto d_qubits_uptr = cutf::cuda::memory::get_device_unique_ptr<qubit_t>(N);

	// 発行命令列
	std::vector<inst_t> insts_vec;

	// 読み取り
	for(std::size_t k_index = 0; k_index < k; k_index++){
		char gate[4];
		// 命令種別読み取り
		std::scanf("%s", gate);

		// 解析
		if(gate[0] == 'X' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts_vec.push_back(inst_type_x<<61 | (static_cast<inst_t>(1)<<target));
		}else if(gate[0] == 'Z' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts_vec.push_back(inst_type_z<<61 | (static_cast<inst_t>(1)<<target));
		}else if(gate[0] == 'H' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts_vec.push_back(inst_type_h<<61 | (static_cast<inst_t>(1)<<target));
		}else if(gate[0] == 'C' && gate[1] == 'X' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			insts_vec.push_back(inst_type_cx<<61 | (static_cast<inst_t>(ctrl) << 32) | (static_cast<inst_t>(1)<<target));
		}else if(gate[0] == 'C' && gate[1] == 'Z' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			insts_vec.push_back(inst_type_cz<<61 | (static_cast<inst_t>(ctrl) << 32) | (static_cast<inst_t>(1)<<target));
		}else if(gate[0] == 'C' && gate[1] == 'C' && gate[2] == 'X' && gate[3] == '\0'){
			std::size_t target, ctrl_0, ctrl_1;
			std::scanf("%lu%lu%lu", &ctrl_0, &ctrl_1, &target);
			insts_vec.push_back(inst_type_ccx<<61 | (static_cast<inst_t>(ctrl_1) << 37) | (static_cast<inst_t>(ctrl_0) << 32) | (static_cast<inst_t>(1)<<target));
		}
	}
#ifdef DEBUG
	debug_print_insts(insts_vec);
#endif

	const std::size_t num_insts = insts_vec.size();
	// 命令列 on デバイスメモリ
	// TODO : 本当はConstantメモリに載せたい
	auto d_insts_uptr = cutf::cuda::memory::get_device_unique_ptr<inst_t>(num_insts);
	cutf::cuda::memory::copy(d_insts_uptr.get(), insts_vec.data(), num_insts);

#ifdef DEBUG
	std::cout<<"start simulation"<<std::endl;
#endif
	// cooperative_groupsでthis_gridを使うので，Launchを手動で行う
	const dim3 grid((N + num_threads_per_block - 1) / num_threads_per_block);
	const dim3 block(num_threads_per_block);
	const auto d_qubits_ptr = d_qubits_uptr.get();
	const auto d_insts_ptr = d_insts_uptr.get();
	const void* args[] = {
		reinterpret_cast<void* const*>(&d_qubits_ptr), reinterpret_cast<void* const*>(&d_insts_ptr), reinterpret_cast<const void*>(&num_insts), reinterpret_cast<const void*>(&N), nullptr
	};
	cutf::cuda::error::check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(qusimu_kernel), grid, block, (void**)args), __FILE__, __LINE__, __func__);
	cudaDeviceSynchronize();
#ifdef DEBUG
	std::cout<<"done"<<std::endl;
#endif

	// 最大のものを取り出す
	auto d_max_uptr = cutf::cuda::memory::get_device_unique_ptr<qubit_t>(1);
	float h_max = 0.0f;
	cutf::cuda::memory::copy(d_max_uptr.get(), &h_max, 1);
	maxabs<<<(N + num_threads_per_block - 1)/num_threads_per_block, num_threads_per_block>>>(d_qubits_uptr.get(), d_max_uptr.get());
	cutf::cuda::memory::copy(&h_max, d_max_uptr.get(), 1);
	printf("%e\n", h_max * h_max);
}
