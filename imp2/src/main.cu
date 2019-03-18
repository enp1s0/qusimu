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
constexpr std::size_t num_threads_per_block = 1 << 8;

// 命令は固定長
using inst_t = uint64_t;
using inst_type_t = uint64_t;
using qubit_t = float;
// 命令塊 (64bitに18bit命令を3命令詰め込む)
// |53      |35      |17      |
// | inst 2 | inst 1 | inst 0 |
// 命令 (18bit)
// |17      15|14           10|9            5 |4          0|
// | 命令種別 | コントロール1 | コントロール0 | ターゲット |

// 命令種別 (3 bit)
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
__host__ __device__ void debug_print_inst(const inst_t inst, const std::size_t inst_num = 0){
	const auto target_inst = (inst >> (inst_num * 18)) & 0x3ffff;
	printf("/*0x%016lx*/ ", inst);
	const auto inst_type = target_inst >> 15;
	const auto control_1 = (target_inst >> 10) & 0x1f;
	const auto control_0 = (target_inst >> 5) & 0x1f;
	const auto target = target_inst & 0x1f;
	if(inst_type == inst_type_x) printf("X %lu", target);
	if(inst_type == inst_type_z) printf("Z %lu", target);
	if(inst_type == inst_type_h) printf("H %lu", target);
	if(inst_type == inst_type_cx) printf("CX %lu %lu", control_0, target);
	if(inst_type == inst_type_cz) printf("CZ %lu %lu", control_0, target);
	if(inst_type == inst_type_ccx) printf("CCX %lu %lu %lu", control_1, control_0, target);
	printf("\n");
}
void debug_print_insts(const inst_t* const insts, const std::size_t num_insts){
	printf("loaded instructions\n");
	printf("line /*  hex inst code   */ | decoded code |\n");
	printf("--------------------------------------------\n");
	for(std::size_t i = 0; i < num_insts; i++){
		printf("%4lu ", i);
		debug_print_inst(insts[i/3], 2 - i%3);
	}
}
inst_t make_inst(const inst_type_t type, const std::size_t control_1, const std::size_t control_0, const std::size_t target){
	return (type << 15) | (control_1 << 10) | (control_0 << 5) | target;
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

__device__ void convert_x(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t target_bits, const std::size_t tid, const std::size_t num_all_threads){
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
__device__ void convert_z(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t target_bits, const std::size_t tid, const std::size_t num_all_threads){
	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		if((i0 & target_bits) != 0)
			qubits[i0] = -qubits[i0];
		else
			qubits[i1] = -qubits[i1];
	}
}
__device__ void convert_h(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t target_bits, const std::size_t tid, const std::size_t num_all_threads){
	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		// NOTE: for内で値のvalidationをしているのでここでは省略
		const auto p0 = qubits[i0];
		const auto p1 = qubits[i1];
		if((i0 & target_bits) == 0){
			qubits[i0] = (p0 + p1) * rsqrt2;
			qubits[i1] = (p0 - p1) * rsqrt2;
		}else{
			qubits[i0] = (p1 - p0) * rsqrt2;
			qubits[i1] = (p1 + p0) * rsqrt2;
		}
	}
}
__device__ void convert_cx(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t ctrl_bits, const std::size_t target_bits, const std::size_t tid, const std::size_t num_all_threads){
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
__device__ void convert_cz(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t ctrl_bits, const std::size_t target_bits, const std::size_t tid, const std::size_t num_all_threads){
	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		if((i0 & ctrl_bits) == 0){
			continue;
		}

		if((i0 & target_bits) != 0)
			qubits[i0] = -qubits[i0];
		else
			qubits[i1] = -qubits[i1];
	}
}
__device__ void convert_ccx(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t ctrl_bits_0, const std::size_t ctrl_bits_1, const std::size_t target_bits, const std::size_t tid, const std::size_t num_all_threads){
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
	for(std::size_t inst_index = 0; inst_index < num_insts/3; inst_index++){
		const auto packed_inst = instruction_array[inst_index];
		// 3つ固まっているうち何番目を使うか
#pragma unroll
		for(std::size_t packed_inst_index = 0; packed_inst_index < 3 ; packed_inst_index++){
			if((inst_index * 3 + packed_inst_index) >= num_insts) return;
			all_threads_group.sync();
			// デコード
			// packed_inst_index番目を取り出す
			const auto inst = (packed_inst >> (18 * (2 - packed_inst_index))) & 0x3ffff;
			//if(tid == 0) debug_print_inst(inst, 0); // 切り出したものなので常に0番目
			//continue;
			// |17    15|が命令種別なのでマジックナンバー15
			const auto inst_type = static_cast<inst_type_t>(inst >> 15);
			const auto target_bits = static_cast<inst_t>(1) << (inst & 0x1f);

			// X
			if(inst_type == inst_type_x){
				convert_x(qubits, num_qubits, target_bits, tid, num_all_threads);
				continue;
			}

			// Z
			if(inst_type == inst_type_z){
				convert_z(qubits, num_qubits, target_bits, tid, num_all_threads);
				continue;
			}

			// H
			if(inst_type == inst_type_h){
				convert_h(qubits, num_qubits, target_bits, tid, num_all_threads);
				continue;
			}

			const auto ctrl_bits_0 = static_cast<inst_t>(1) << ((inst >> 5) & 0x1f);
			// CX
			if(inst_type == inst_type_cx){
				convert_cx(qubits, num_qubits, ctrl_bits_0, target_bits, tid, num_all_threads);
				continue;
			}

			// CZ
			if(inst_type == inst_type_cz){
				convert_cz(qubits, num_qubits, ctrl_bits_0, target_bits, tid, num_all_threads);
				continue;
			}

			const auto ctrl_bits_1 = static_cast<inst_t>(1) << ((inst >> 10) & 0x1f);
			// CCX
			if(inst_type == inst_type_ccx){
				convert_ccx(qubits, num_qubits, ctrl_bits_0, ctrl_bits_1, target_bits, tid, num_all_threads);
				continue;
			}
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
	std::size_t n, num_insts;
	std::cin >> n >> num_insts;

	// 量子ビットの組み合わせ総数
	const std::size_t N = 1 << n;

	// 量子ビット on デバイスメモリ
	auto d_qubits_uptr = cutf::cuda::memory::get_device_unique_ptr<qubit_t>(N);

	// 発行命令列
	inst_t insts[15000];
	std::size_t inst_index = 0;

	// 読み取り
	std::size_t k_index = 0;
	for(; k_index < num_insts; k_index++){
		char gate[4];
		// 命令種別読み取り
		std::scanf("%s", gate);

		// 解析
		if(gate[0] == 'X' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts[inst_index] |= make_inst(inst_type_x, 0, 0, target);
		}else if(gate[0] == 'Z' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts[inst_index] |= make_inst(inst_type_z, 0, 0, target);
		}else if(gate[0] == 'H' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			insts[inst_index] |= make_inst(inst_type_h, 0, 0, target);
		}else if(gate[0] == 'C' && gate[1] == 'X' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			insts[inst_index] |= make_inst(inst_type_cx, 0, ctrl, target);
		}else if(gate[0] == 'C' && gate[1] == 'Z' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			insts[inst_index] |= make_inst(inst_type_cz, 0, ctrl, target);
		}else if(gate[0] == 'C' && gate[1] == 'C' && gate[2] == 'X' && gate[3] == '\0'){
			std::size_t target, ctrl_0, ctrl_1;
			std::scanf("%lu%lu%lu", &ctrl_1, &ctrl_0, &target);
			insts[inst_index] |= make_inst(inst_type_ccx, ctrl_1, ctrl_0, target);
		}
		if(k_index % 3 == 2){
			inst_index++;
		}else{
			insts[inst_index] <<= 18;
		}
	}
	// ループ内でinst_index++しているから，配列サイズ(固定値)を変更すると範囲外アクセスの危険大
	// そのため事前に判定しておく
	if(k_index % 3 == 0){
		insts[inst_index] <<= 36;
	}else if(k_index % 3 == 1){
		insts[inst_index] <<= 18;
	}
	// 命令列 on デバイスメモリ
	// TODO : 本当はConstantメモリに載せたい
	cudaMemcpyToSymbol(instruction_array, insts, num_insts * sizeof(inst_t));
	// Occupansyが最大になるblock数を取得
	const auto device_list = cutf::cuda::device::get_properties_vector();
	int num_blocks_0 = device_list[0].multiProcessorCount;
	int num_blocks_1;
	cutf::cuda::error::check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_1, qusimu_kernel, num_threads_per_block, 0), __FILE__, __LINE__, __func__);
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
	   	reinterpret_cast<const void*>(&num_insts),
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
