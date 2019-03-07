#include <iostream>
#include <string>
#include <vector>
#include <cooperative_groups.h>
#include <cutf/memory.hpp>
#include <cutf/device.hpp>

// CUDAの組み込み関数はconstexprではないので
constexpr float rsqrt2 = 1.0f/1.41421356237f;
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

// 命令はconstant memoryに乗せておく
// 取り敢えず40kB分確保
__constant__ inst_t instruction_array[7 * 1024];

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
void debug_print_insts(const inst_t* const insts, const std::size_t num_insts){
	printf("loaded instructions\n");
	printf("line /*  hex inst code   */ | decoded code |\n");
	printf("--------------------------------------------\n");
	for(std::size_t i = 0; i < num_insts; i++){
		printf("%4lu ", i);
		debug_print_inst(insts[i]);
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

__device__ void init_qubits(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t tid, const std::size_t num_all_threads){
	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		qubits[index] = static_cast<qubit_t>(0);
	}
	if(tid == 0) qubits[0] = static_cast<qubit_t>(1);
}

__device__ void convert_x(qubit_t* const qubits, const std::size_t num_qubits, const inst_t inst, const std::size_t tid, const std::size_t num_all_threads){
	// 交換部分の解析
	constexpr auto target_mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & target_mask;

	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		// NOTE: for内で値のvalidationをしているのでここでは省略
		const auto p0 = qubits[i0];
		const auto p1 = qubits[i1];
		qubits[i0] = p1;
		qubits[i1] = p0;
	}
}
__device__ void convert_z(qubit_t* const qubits, const std::size_t num_qubits, const inst_t inst, const std::size_t tid, const std::size_t num_all_threads){
	// 交換部分の解析
	constexpr auto target_mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & target_mask;

	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		if((i0 & target_bits) == 0)
			qubits[i0] = -qubits[i0];
		else
			qubits[i1] = -qubits[i1];
	}
}
__device__ void convert_h(qubit_t* const qubits, const std::size_t num_qubits, const inst_t inst, const std::size_t tid, const std::size_t num_all_threads){
	// 交換部分の解析
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;

	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		// NOTE: for内で値のvalidationをしているのでここでは省略
		const auto p0 = qubits[i0];
		const auto p1 = qubits[i1];
		if((i0 & target_bits) == 0){
			qubits[i0] = (p0 + p1) / sqrt2;
			qubits[i1] = (p0 - p1) / sqrt2;
		}else{
			qubits[i0] = (p1 - p0) / sqrt2;
			qubits[i1] = (p1 + p0) / sqrt2;
		}
	}
}
__device__ void convert_cx(qubit_t* const qubits, const std::size_t num_qubits, const inst_t inst, const std::size_t tid, const std::size_t num_all_threads){
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits = static_cast<inst_t>(1) << ((inst >> 32) & 0x1f);

	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		if((i0 & ctrl_bits) == 0){
			continue;
		}

		// NOTE: for内で値のvalidationをしているのでここでは省略
		const auto p0 = qubits[i0];
		const auto p1 = qubits[i1];
		qubits[i0] = p1;
		qubits[i1] = p0;
	}
}
__device__ void convert_cz(qubit_t* const qubits, const std::size_t num_qubits, const inst_t inst, const std::size_t tid, const std::size_t num_all_threads){
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits = static_cast<inst_t>(1) << ((inst >> 32) & 0x1f);

	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		if((index & ctrl_bits) == 0){
			continue;
		}

		if((i0 & target_bits) == 0)
			qubits[i0] = -qubits[i0];
		else
			qubits[i1] = -qubits[i1];
	}
}
__device__ void convert_ccx(qubit_t* const qubits, const std::size_t num_qubits, const inst_t inst, const std::size_t tid, const std::size_t num_all_threads){
	constexpr auto mask = ((static_cast<inst_t>(1)<<31) - 1);
	const auto target_bits = inst & mask;
	// 31bit目から5bitがcontrolなので
	const auto ctrl_bits_0 = static_cast<inst_t>(1) << ((inst >> 32) & 0x1f);
	const auto ctrl_bits_1 = static_cast<inst_t>(1) << ((inst >> 37) & 0x1f);

	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		if(((i0 & ctrl_bits_0) == 0) || ((i0 & ctrl_bits_1) == 0)){
			continue;
		}

		// NOTE: for内で値のvalidationをしているのでここでは省略
		const auto p0 = qubits[i0];
		const auto p1 = qubits[i1];
		qubits[i0] = p1;
		qubits[i1] = p0;
	}
}

__global__ void qusimu_kernel(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t num_insts, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 初期化
	init_qubits(qubits, num_qubits, tid, num_all_threads);
	// 全スレッドでgroupを作る
	const auto all_threads_group = cooperative_groups::this_grid();
	// 命令実行ループ
	for(std::size_t inst_index = 0; inst_index < num_insts; inst_index++){
		all_threads_group.sync();
		// デコード
		const auto inst = instruction_array[inst_index];
		//if(tid == 0) debug_print_inst(inst);
		// |63   61|が命令種別なのでマジックナンバー61
		const auto inst_type = static_cast<inst_type_t>(inst >> 61);

		// X
		if(inst_type == inst_type_x){
			convert_x(qubits, num_qubits, inst, tid, num_all_threads);
			continue;
		}

		// Z
		if(inst_type == inst_type_z){
			convert_z(qubits, num_qubits, inst, tid, num_all_threads);
			continue;
		}

		// H
		if(inst_type == inst_type_h){
			convert_h(qubits, num_qubits, inst, tid, num_all_threads);
			continue;
		}

		// CX
		if(inst_type == inst_type_cx){
			convert_cx(qubits, num_qubits, inst, tid, num_all_threads);
			continue;
		}

		// CZ
		if(inst_type == inst_type_cz){
			convert_cz(qubits, num_qubits, inst, tid, num_all_threads);
			continue;
		}

		// CCX
		if(inst_type == inst_type_ccx){
			convert_ccx(qubits, num_qubits, inst, tid, num_all_threads);
			continue;
		}
	}
	// sync all test
	/*for(std::size_t i = 0; i < num_all_threads; i++){
		all_threads_group.sync();
		if(i == tid){
			printf("%lu \n", i);
		}
	}*/
}

int main(){
	std::size_t n, k;
	std::cin >> n >> k;

	// 量子ビットの組み合わせ総数
	const std::size_t N = 1 << n;

	// 量子ビット on デバイスメモリ
	auto d_qubits_uptr = cutf::cuda::memory::get_device_unique_ptr<qubit_t>(N);

	// 発行命令列
	inst_t insts[5000];

	// 読み取り
	for(std::size_t k_index = 0; k_index < k; k_index++){
		char gate[4];
		// 命令種別読み取り
		std::scanf("%s", gate);

		// 解析
		if(gate[0] == 'X' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts[k_index] = inst_type_x<<61 | (static_cast<inst_t>(1)<<target);
		}else if(gate[0] == 'Z' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts[k_index] = inst_type_z<<61 | (static_cast<inst_t>(1)<<target);
		}else if(gate[0] == 'H' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts[k_index] = inst_type_h<<61 | (static_cast<inst_t>(1)<<target);
		}else if(gate[0] == 'C' && gate[1] == 'X' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			insts[k_index] = inst_type_cx<<61 | (static_cast<inst_t>(ctrl) << 32) | (static_cast<inst_t>(1)<<target);
		}else if(gate[0] == 'C' && gate[1] == 'Z' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			insts[k_index] = inst_type_cz<<61 | (static_cast<inst_t>(ctrl) << 32) | (static_cast<inst_t>(1)<<target);
		}else if(gate[0] == 'C' && gate[1] == 'C' && gate[2] == 'X' && gate[3] == '\0'){
			std::size_t target, ctrl_0, ctrl_1;
			std::scanf("%lu%lu%lu", &ctrl_0, &ctrl_1, &target);
			insts[k_index] = inst_type_ccx<<61 | (static_cast<inst_t>(ctrl_1) << 37) | (static_cast<inst_t>(ctrl_0) << 32) | (static_cast<inst_t>(1)<<target);
		}
	}

	// 命令列 on デバイスメモリ
	// TODO : 本当はConstantメモリに載せたい
	cudaMemcpyToSymbol(instruction_array, insts, k * sizeof(inst_t));
	// Occupansyが最大になるblock数を取得
	const auto device_list = cutf::cuda::device::get_properties_vector();
	int num_blocks_0 = device_list[0].multiProcessorCount;
	int num_blocks_1;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_1, qusimu_kernel, num_threads_per_block, 0);
	int num_blocks = num_blocks_0 * num_blocks_1;
	std::cout<<"Grid size  : "<<num_blocks<<std::endl;
	std::cout<<"Block size : "<<num_threads_per_block<<std::endl;
	const std::size_t num_all_threads = num_blocks * num_threads_per_block;
	
	// cooperative_groupsでthis_gridを使うので，Launchを手動で行う
	const dim3 grid(num_blocks);
	const dim3 block(num_threads_per_block);
	const auto d_qubits_ptr = d_qubits_uptr.get();
	const void* args[] = {
		reinterpret_cast<void* const*>(&d_qubits_ptr),
	   	reinterpret_cast<const void*>(&N),
	   	reinterpret_cast<const void*>(&k),
		reinterpret_cast<const void*>(&num_all_threads),
	   	nullptr
	};
	cutf::cuda::error::check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(qusimu_kernel), grid, block, (void**)args), __FILE__, __LINE__, __func__);
	cudaDeviceSynchronize();

	// 最大のものを取り出す
	/*
	auto d_max_uptr = cutf::cuda::memory::get_device_unique_ptr<qubit_t>(1);
	float h_max = 0.0f;
	cutf::cuda::memory::copy(d_max_uptr.get(), &h_max, 1);
	maxabs<<<(N + num_threads_per_block - 1)/num_threads_per_block, num_threads_per_block>>>(d_qubits_uptr.get(), d_max_uptr.get());
	cutf::cuda::memory::copy(&h_max, d_max_uptr.get(), 1);
	printf("%e\n", h_max * h_max);
	*/
	auto h_qubits_uptr = cutf::cuda::memory::get_host_unique_ptr<qubit_t>(N);
	cutf::cuda::memory::copy(h_qubits_uptr.get(), d_qubits_uptr.get(), N);
	std::size_t max_i;
	qubit_t max_p = 0;
	for(std::size_t i = 0; i < N; i++){
		//printf("[%8lu] : %.8f\n", i, h_qubits_uptr.get()[i]);
		const auto p = h_qubits_uptr.get()[i];
		if(p * p > max_p * max_p){
			max_p = p;
			max_i = i;
		}
	}
	printf("%lu\n%.8e\n", max_i, (max_p * max_p));
}
