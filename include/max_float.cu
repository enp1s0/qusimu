#define WARPSIZE 32

__device__ float warpReduceMax(float val){
  for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
#if __CUDACC_VER_MAJOR__ >= 9
    val = fmaxf(val, __shfl_down_sync(~0, val, offset));
#else
    val = fmaxf(val, __shfl_down(val, offset));
#endif
  return val;
}

__global__ void maxabs(float *A, float *m){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int lane = threadIdx.x % WARPSIZE;
  float val = fabsf(A[i]);
  val = warpReduceMax(val);
  if(lane == 0) atomicMax((int *) m, *(int *) &val);
}
