
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long

extern "C" {

__device__ __forceinline__ void AsmBlockSync(int name, int numThreads) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}


__device__ __forceinline__ void AsmWarpSync(const unsigned threadsmask) {
  asm volatile("bar.warp.sync %0;" : : "r"(threadsmask) : "memory");
}

__device__ __forceinline__ void default_function_kernel0_block_0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT, char* shared_buffer, const uint& block_idx, const uint& thread_idx) {
  if (thread_idx >= 32){
    return;
  }
  const dim3 gridDim(256, 1, 1);
  const dim3 blockDim(32, 1, 1);

  const dim3 blockIdx(block_idx);
  float* placeholder_d_shared = (float*)(shared_buffer + 0);
  float* placeholder_shared = (float*)(shared_buffer + 4096);

  float T_batch_matmul_NT_local[4];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  T_batch_matmul_NT_local[2] = 0.000000e+00f;
  T_batch_matmul_NT_local[3] = 0.000000e+00f;
  *(float2*)(placeholder_d_shared + (((int)threadIdx.x) * 2)) = *(float2*)(placeholder + ((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 64)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 64));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 128));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 192)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 192));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 256));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 320)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 320));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 384)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 384));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 448)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 448));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4096));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 576)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4160));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 640)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4224));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 704)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4288));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4352));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 832)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4416));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 896)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4480));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 960)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4544));
  placeholder_shared[((int)threadIdx.x)] = placeholder1[((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x))];
  placeholder_shared[(((int)threadIdx.x) + 32)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 32)];
  placeholder_shared[(((int)threadIdx.x) + 64)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 64)];
  placeholder_shared[(((int)threadIdx.x) + 96)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 96)];
  placeholder_shared[(((int)threadIdx.x) + 128)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 128)];
  placeholder_shared[(((int)threadIdx.x) + 160)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 160)];
  placeholder_shared[(((int)threadIdx.x) + 192)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 192)];
  placeholder_shared[(((int)threadIdx.x) + 224)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 224)];
  placeholder_shared[(((int)threadIdx.x) + 256)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 256)];
  placeholder_shared[(((int)threadIdx.x) + 288)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 288)];
  placeholder_shared[(((int)threadIdx.x) + 320)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 320)];
  placeholder_shared[(((int)threadIdx.x) + 352)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 352)];
  placeholder_shared[(((int)threadIdx.x) + 384)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 384)];
  placeholder_shared[(((int)threadIdx.x) + 416)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 416)];
  placeholder_shared[(((int)threadIdx.x) + 448)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 448)];
  placeholder_shared[(((int)threadIdx.x) + 480)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 480)];
  placeholder_shared[(((int)threadIdx.x) + 512)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4096)];
  placeholder_shared[(((int)threadIdx.x) + 544)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4128)];
  placeholder_shared[(((int)threadIdx.x) + 576)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4160)];
  placeholder_shared[(((int)threadIdx.x) + 608)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4192)];
  placeholder_shared[(((int)threadIdx.x) + 640)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4224)];
  placeholder_shared[(((int)threadIdx.x) + 672)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4256)];
  placeholder_shared[(((int)threadIdx.x) + 704)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4288)];
  placeholder_shared[(((int)threadIdx.x) + 736)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4320)];
  placeholder_shared[(((int)threadIdx.x) + 768)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4352)];
  placeholder_shared[(((int)threadIdx.x) + 800)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4384)];
  placeholder_shared[(((int)threadIdx.x) + 832)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4416)];
  placeholder_shared[(((int)threadIdx.x) + 864)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4448)];
  placeholder_shared[(((int)threadIdx.x) + 896)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4480)];
  placeholder_shared[(((int)threadIdx.x) + 928)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4512)];
  placeholder_shared[(((int)threadIdx.x) + 960)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4544)];
  placeholder_shared[(((int)threadIdx.x) + 992)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4576)];
  AsmBlockSync(0, 32);
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[(((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256))]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 64)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 128)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 192)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 1)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 65)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 129)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 193)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 2)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 66)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 130)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 194)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 3)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 67)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 131)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 195)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 4)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 68)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 132)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 196)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 5)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 69)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 133)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 197)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 6)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 70)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 134)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 198)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 7)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 71)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 135)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 199)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 8)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 72)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 136)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 200)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 9)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 73)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 137)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 201)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 10)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 74)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 138)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 202)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 11)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 75)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 139)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 203)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 12)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 76)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 140)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 204)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 13)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 77)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 141)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 205)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 14)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 78)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 142)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 206)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 15)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 79)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 143)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 207)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 16)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 80)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 144)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 208)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 17)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 81)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 145)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 209)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 18)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 82)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 146)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 210)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 19)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 83)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 147)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 211)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 20)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 84)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 148)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 212)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 21)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 85)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 149)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 213)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 22)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 86)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 150)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 214)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 23)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 87)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 151)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 215)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 24)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 88)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 152)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 216)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 25)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 89)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 153)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 217)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 26)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 90)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 154)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 218)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 27)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 91)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 155)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 219)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 28)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 92)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 156)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 220)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 29)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 93)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 157)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 221)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 30)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 94)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 158)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 222)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 31)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 95)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 159)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 223)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 32)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 96)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 160)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 224)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 33)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 97)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 161)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 225)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 34)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 98)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 162)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 226)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 35)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 99)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 163)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 227)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 36)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 100)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 164)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 228)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 37)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 101)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 165)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 229)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 38)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 102)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 166)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 230)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 39)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 103)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 167)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 231)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 40)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 104)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 168)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 232)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 41)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 105)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 169)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 233)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 42)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 106)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 170)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 234)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 43)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 107)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 171)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 235)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 44)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 108)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 172)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 236)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 45)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 109)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 173)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 237)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 46)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 110)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 174)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 238)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 47)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 111)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 175)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 239)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 48)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 112)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 176)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 240)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 49)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 113)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 177)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 241)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 50)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 114)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 178)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 242)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 51)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 115)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 179)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 243)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 52)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 116)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 180)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 244)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 53)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 117)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 181)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 245)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 54)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 118)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 182)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 246)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 55)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 119)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 183)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 247)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 56)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 120)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 184)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 248)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 57)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 121)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 185)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 249)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 58)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 122)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 186)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 250)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 59)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 123)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 187)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 251)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 60)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 124)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 188)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 252)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 61)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 125)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 189)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 253)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 62)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 126)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 190)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 254)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 63)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 127)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 191)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 255)]));
  for (int j_inner = 0; j_inner < 4; ++j_inner) {
    T_batch_matmul_NT[((((((((((int)blockIdx.x) >> 6) * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((((int)threadIdx.x) & 15) >> 1) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + j_inner)] = T_batch_matmul_NT_local[j_inner];
  }
}

__device__ __forceinline__ void default_function_kernel0_block_1(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT, char* shared_buffer, const uint& block_idx, const uint& thread_idx) {
  if (thread_idx >= 32){
    return;
  }
  const dim3 gridDim(256, 1, 1);
  const dim3 blockDim(32, 1, 1);

  const dim3 blockIdx(block_idx);
  float* placeholder_d_shared = (float*)(shared_buffer + 0);
  float* placeholder_shared = (float*)(shared_buffer + 4096);

  float T_batch_matmul_NT_local[4];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  T_batch_matmul_NT_local[2] = 0.000000e+00f;
  T_batch_matmul_NT_local[3] = 0.000000e+00f;
  *(float2*)(placeholder_d_shared + (((int)threadIdx.x) * 2)) = *(float2*)(placeholder + ((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 64)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 64));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 128));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 192)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 192));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 256));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 320)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 320));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 384)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 384));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 448)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 448));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4096));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 576)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4160));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 640)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4224));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 704)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4288));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4352));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 832)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4416));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 896)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4480));
  *(float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2) + 960)) = *(float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((int)threadIdx.x) * 2)) + 4544));
  placeholder_shared[((int)threadIdx.x)] = placeholder1[((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x))];
  placeholder_shared[(((int)threadIdx.x) + 32)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 32)];
  placeholder_shared[(((int)threadIdx.x) + 64)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 64)];
  placeholder_shared[(((int)threadIdx.x) + 96)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 96)];
  placeholder_shared[(((int)threadIdx.x) + 128)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 128)];
  placeholder_shared[(((int)threadIdx.x) + 160)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 160)];
  placeholder_shared[(((int)threadIdx.x) + 192)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 192)];
  placeholder_shared[(((int)threadIdx.x) + 224)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 224)];
  placeholder_shared[(((int)threadIdx.x) + 256)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 256)];
  placeholder_shared[(((int)threadIdx.x) + 288)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 288)];
  placeholder_shared[(((int)threadIdx.x) + 320)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 320)];
  placeholder_shared[(((int)threadIdx.x) + 352)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 352)];
  placeholder_shared[(((int)threadIdx.x) + 384)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 384)];
  placeholder_shared[(((int)threadIdx.x) + 416)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 416)];
  placeholder_shared[(((int)threadIdx.x) + 448)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 448)];
  placeholder_shared[(((int)threadIdx.x) + 480)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 480)];
  placeholder_shared[(((int)threadIdx.x) + 512)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4096)];
  placeholder_shared[(((int)threadIdx.x) + 544)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4128)];
  placeholder_shared[(((int)threadIdx.x) + 576)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4160)];
  placeholder_shared[(((int)threadIdx.x) + 608)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4192)];
  placeholder_shared[(((int)threadIdx.x) + 640)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4224)];
  placeholder_shared[(((int)threadIdx.x) + 672)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4256)];
  placeholder_shared[(((int)threadIdx.x) + 704)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4288)];
  placeholder_shared[(((int)threadIdx.x) + 736)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4320)];
  placeholder_shared[(((int)threadIdx.x) + 768)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4352)];
  placeholder_shared[(((int)threadIdx.x) + 800)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4384)];
  placeholder_shared[(((int)threadIdx.x) + 832)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4416)];
  placeholder_shared[(((int)threadIdx.x) + 864)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4448)];
  placeholder_shared[(((int)threadIdx.x) + 896)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4480)];
  placeholder_shared[(((int)threadIdx.x) + 928)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4512)];
  placeholder_shared[(((int)threadIdx.x) + 960)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4544)];
  placeholder_shared[(((int)threadIdx.x) + 992)] = placeholder1[(((((((int)blockIdx.x) >> 6) * 8192) + ((((int)blockIdx.x) & 7) * 512)) + ((int)threadIdx.x)) + 4576)];
  AsmBlockSync(1, 32);
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[(((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256))]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 64)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 128)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[((((int)threadIdx.x) >> 1) * 64)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 192)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 1)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 65)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 129)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 1)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 193)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 2)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 66)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 130)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 2)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 194)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 3)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 67)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 131)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 3)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 195)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 4)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 68)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 132)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 4)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 196)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 5)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 69)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 133)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 5)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 197)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 6)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 70)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 134)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 6)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 198)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 7)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 71)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 135)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 7)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 199)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 8)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 72)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 136)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 8)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 200)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 9)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 73)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 137)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 9)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 201)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 10)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 74)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 138)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 10)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 202)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 11)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 75)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 139)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 11)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 203)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 12)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 76)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 140)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 12)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 204)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 13)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 77)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 141)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 13)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 205)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 14)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 78)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 142)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 14)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 206)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 15)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 79)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 143)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 15)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 207)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 16)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 80)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 144)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 16)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 208)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 17)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 81)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 145)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 17)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 209)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 18)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 82)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 146)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 18)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 210)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 19)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 83)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 147)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 19)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 211)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 20)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 84)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 148)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 20)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 212)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 21)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 85)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 149)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 21)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 213)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 22)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 86)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 150)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 22)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 214)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 23)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 87)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 151)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 23)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 215)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 24)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 88)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 152)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 24)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 216)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 25)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 89)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 153)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 25)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 217)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 26)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 90)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 154)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 26)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 218)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 27)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 91)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 155)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 27)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 219)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 28)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 92)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 156)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 28)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 220)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 29)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 93)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 157)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 29)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 221)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 30)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 94)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 158)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 30)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 222)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 31)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 95)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 159)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 31)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 223)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 32)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 96)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 160)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 32)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 224)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 33)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 97)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 161)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 33)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 225)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 34)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 98)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 162)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 34)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 226)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 35)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 99)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 163)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 35)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 227)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 36)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 100)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 164)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 36)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 228)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 37)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 101)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 165)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 37)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 229)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 38)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 102)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 166)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 38)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 230)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 39)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 103)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 167)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 39)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 231)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 40)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 104)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 168)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 40)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 232)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 41)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 105)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 169)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 41)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 233)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 42)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 106)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 170)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 42)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 234)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 43)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 107)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 171)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 43)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 235)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 44)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 108)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 172)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 44)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 236)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 45)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 109)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 173)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 45)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 237)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 46)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 110)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 174)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 46)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 238)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 47)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 111)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 175)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 47)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 239)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 48)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 112)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 176)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 48)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 240)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 49)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 113)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 177)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 49)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 241)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 50)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 114)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 178)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 50)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 242)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 51)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 115)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 179)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 51)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 243)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 52)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 116)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 180)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 52)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 244)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 53)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 117)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 181)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 53)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 245)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 54)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 118)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 182)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 54)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 246)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 55)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 119)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 183)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 55)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 247)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 56)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 120)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 184)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 56)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 248)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 57)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 121)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 185)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 57)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 249)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 58)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 122)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 186)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 58)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 250)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 59)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 123)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 187)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 59)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 251)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 60)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 124)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 188)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 60)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 252)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 61)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 125)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 189)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 61)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 253)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 62)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 126)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 190)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 62)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 254)]));
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 63)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 127)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 191)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 64) + 63)] * placeholder_shared[((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 1) * 256)) + 255)]));
  for (int j_inner = 0; j_inner < 4; ++j_inner) {
    T_batch_matmul_NT[((((((((((int)blockIdx.x) >> 6) * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((((int)blockIdx.x) & 63) >> 3) * 512)) + (((((int)threadIdx.x) & 15) >> 1) * 64)) + ((((int)blockIdx.x) & 7) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + j_inner)] = T_batch_matmul_NT_local[j_inner];
  }
}

__global__ void __launch_bounds__(32) sample(float* __restrict__ placeholder[], float* __restrict__ T_batch_matmul_NT[], float* __restrict__ placeholder1[], uint sample_0_capacity, uint sample_1_capacity, uint sample_2_capacity, uint sample_3_capacity) {
  // [kernel_type] homo_fuse
  // [thread_extent] blockIdx.xdim = [256, 256]
  // [thread_extent] blockIdx.ydim = 1
  // [thread_extent] blockIdx.zdim = 1
  // [thread_extent] threadIdx.xdim = [32, 32]
  // [thread_extent] threadIdx.ydim = 1
  // [thread_extent] threadIdx.zdim = 1

  int capacity_dims[2];
  capacity_dims[0] = 256;
  capacity_dims[1] = 256;
  __shared__ char shared_buffer[8192];
  if (blockIdx.x < (capacity_dims[sample_0_capacity])) {
    switch (sample_0_capacity) {
      case 0: {
        default_function_kernel0_block_0(placeholder[0], T_batch_matmul_NT[0], placeholder1[0], shared_buffer, blockIdx.x - 0, threadIdx.x);
      }
      case 1: {
        default_function_kernel0_block_1(placeholder[0], T_batch_matmul_NT[0], placeholder1[0], shared_buffer, blockIdx.x - 0, threadIdx.x);
      }
    }
  }
  else if (blockIdx.x < (capacity_dims[sample_0_capacity] + capacity_dims[sample_1_capacity])) {
    switch (sample_1_capacity) {
      case 0: {
        default_function_kernel0_block_0(placeholder[1], T_batch_matmul_NT[1], placeholder1[1], shared_buffer, blockIdx.x - (capacity_dims[sample_0_capacity]), threadIdx.x);
      }
      case 1: {
        default_function_kernel0_block_1(placeholder[1], T_batch_matmul_NT[1], placeholder1[1], shared_buffer, blockIdx.x - (capacity_dims[sample_0_capacity]), threadIdx.x);
      }
    }
  }
  else if (blockIdx.x < (capacity_dims[sample_0_capacity] + capacity_dims[sample_1_capacity] + capacity_dims[sample_2_capacity])) {
    switch (sample_2_capacity) {
      case 0: {
        default_function_kernel0_block_0(placeholder[2], T_batch_matmul_NT[2], placeholder1[2], shared_buffer, blockIdx.x - (capacity_dims[sample_0_capacity] + capacity_dims[sample_1_capacity]), threadIdx.x);
      }
      case 1: {
        default_function_kernel0_block_1(placeholder[2], T_batch_matmul_NT[2], placeholder1[2], shared_buffer, blockIdx.x - (capacity_dims[sample_0_capacity] + capacity_dims[sample_1_capacity]), threadIdx.x);
      }
    }
  }
  else if (blockIdx.x < (capacity_dims[sample_0_capacity] + capacity_dims[sample_1_capacity] + capacity_dims[sample_2_capacity] + capacity_dims[sample_3_capacity])) {
    switch (sample_3_capacity) {
      case 0: {
        default_function_kernel0_block_0(placeholder[3], T_batch_matmul_NT[3], placeholder1[3], shared_buffer, blockIdx.x - (capacity_dims[sample_0_capacity] + capacity_dims[sample_1_capacity] + capacity_dims[sample_2_capacity]), threadIdx.x);
      }
      case 1: {
        default_function_kernel0_block_1(placeholder[3], T_batch_matmul_NT[3], placeholder1[3], shared_buffer, blockIdx.x - (capacity_dims[sample_0_capacity] + capacity_dims[sample_1_capacity] + capacity_dims[sample_2_capacity]), threadIdx.x);
      }
    }
  }
}
} // extern "C"

int main(int argc, char const *argv[])
{
  /* code */
  return 0;
}
