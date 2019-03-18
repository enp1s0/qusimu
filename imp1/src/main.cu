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

__global__ void init_qubits(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		qubits[index] = static_cast<qubit_t>(0);
	}
	if(tid == 0) qubits[0] = static_cast<qubit_t>(1);
}

__global__ void convert_x(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t target_bits, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void convert_z(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t target_bits, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(std::size_t i = 0, index; (index = i + tid) < (num_qubits >> 1); i+= num_all_threads){
		const auto i0 = (index / target_bits) * (target_bits << 1) + (index % target_bits);
		const auto i1 = i0 ^ target_bits;

		if((i0 & target_bits) != 0)
			qubits[i0] = -qubits[i0];
		else
			qubits[i1] = -qubits[i1];
	}
}
__global__ void convert_h(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t target_bits, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void convert_cx(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t ctrl_bits, const std::size_t target_bits, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void convert_cz(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t ctrl_bits, const std::size_t target_bits, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void convert_ccx(qubit_t* const qubits, const std::size_t num_qubits, const std::size_t ctrl_bits_0, const std::size_t ctrl_bits_1, const std::size_t target_bits, const std::size_t num_all_threads){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
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

int main(){
	std::size_t n, num_insts;
	std::cin >> n >> num_insts;

	// 量子ビットの組み合わせ総数
	const std::size_t N = 1 << n;

	// 量子ビット on デバイスメモリ
	auto d_qubits_uptr = cutf::cuda::memory::get_device_unique_ptr<qubit_t>(N);

	// Occupansyが最大になるblock数を取得
	const auto device_list = cutf::cuda::device::get_properties_vector();
	int num_blocks_0 = device_list[0].multiProcessorCount;
	int num_blocks_1;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_1, init_qubits, num_threads_per_block, 0);
	int num_blocks = (N + num_threads_per_block - 1)/num_threads_per_block;
	//int num_blocks = num_blocks_0 * num_blocks_1;
	std::cout<<"Grid size  : "<<num_blocks<<std::endl;
	std::cout<<"Block size : "<<num_threads_per_block<<std::endl;
	const std::size_t num_all_threads = num_blocks * num_threads_per_block;

	// 初期化
	init_qubits<<<num_blocks, num_threads_per_block>>>(d_qubits_uptr.get(), N, num_all_threads);

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
			convert_x<<<num_blocks, num_threads_per_block>>>(d_qubits_uptr.get(), N, (1lu<<target), num_all_threads);
		}else if(gate[0] == 'Z' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			convert_z<<<num_blocks, num_threads_per_block>>>(d_qubits_uptr.get(), N, (1lu<<target), num_all_threads);
		}else if(gate[0] == 'H' && gate[1] == '\0'){
			std::size_t target;
			std::scanf("%lu", &target);
			convert_h<<<num_blocks, num_threads_per_block>>>(d_qubits_uptr.get(), N, (1lu<<target), num_all_threads);
		}else if(gate[0] == 'C' && gate[1] == 'X' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			convert_cx<<<num_blocks, num_threads_per_block>>>(d_qubits_uptr.get(), N, (1lu<<ctrl), (1lu<<target), num_all_threads);
		}else if(gate[0] == 'C' && gate[1] == 'Z' && gate[2] == '\0'){
			std::size_t target, ctrl;
			std::scanf("%lu%lu", &ctrl, &target);
			convert_cz<<<num_blocks, num_threads_per_block>>>(d_qubits_uptr.get(), N, (1lu<<ctrl), (1lu<<target), num_all_threads);
		}else if(gate[0] == 'C' && gate[1] == 'C' && gate[2] == 'X' && gate[3] == '\0'){
			std::size_t target, ctrl_0, ctrl_1;
			std::scanf("%lu%lu%lu", &ctrl_1, &ctrl_0, &target);
			convert_ccx<<<num_blocks, num_threads_per_block>>>(d_qubits_uptr.get(), N, (1lu<<ctrl_0), (1lu<<ctrl_1), (1lu<<target), num_all_threads);
		}
	}
	cudaDeviceSynchronize();

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
