
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long

extern "C" __global__ void __launch_bounds__(56) Conv2dBiasBatchNorm2dReLU__input_0_1_3_224_224__output_0_1_32_112_112__in_channels_3_out_channels_32_kernel_size_7_7_stride_2_2_padding_3_3_dilation_1_1_groups_1(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {
  // [kernel_type] global
  // [thread_extent] blockIdx.x = 896
  // [thread_extent] blockIdx.y = 1
  // [thread_extent] blockIdx.z = 1
  // [thread_extent] threadIdx.x = 56
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.z = 1

  __shared__ float pad_temp_shared[273];
  __shared__ float placeholder_shared[48];

  float conv2d_nchw[8];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  for (int ry_outer_outer = 0; ry_outer_outer < 7; ++ry_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 7; ++rx_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = ((((3 <= (((((((int)blockIdx.x) % 448) >> 4) * 8) + (((int)threadIdx.x) / 13)) + ry_outer_outer)) && (3 <= ((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 13)))) && (((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 13)) < 227)) ? placeholder[(((((((((((int)blockIdx.x) % 448) >> 4) * 1792) + ((((int)threadIdx.x) / 13) * 224)) + (ry_outer_outer * 224)) + ((((int)blockIdx.x) & 15) * 14)) + rx_outer_outer) + (((int)threadIdx.x) % 13)) - 675)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 56)] = (((((3 <= (((((((int)blockIdx.x) % 448) >> 4) * 8) + (((((int)threadIdx.x) + 56) % 91) / 13)) + ry_outer_outer)) && ((((((((int)blockIdx.x) % 448) >> 4) * 8) + (((((int)threadIdx.x) + 56) % 91) / 13)) + ry_outer_outer) < 227)) && (3 <= ((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 13)))) && (((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 13)) < 227)) ? placeholder[((((((((((((int)threadIdx.x) + 56) / 91) * 50176) + (((((int)blockIdx.x) % 448) >> 4) * 1792)) + ((((((int)threadIdx.x) + 56) % 91) / 13) * 224)) + (ry_outer_outer * 224)) + ((((int)blockIdx.x) & 15) * 14)) + rx_outer_outer) + ((((int)threadIdx.x) + 4) % 13)) - 675)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 112)] = (((((3 <= (((((((int)blockIdx.x) % 448) >> 4) * 8) + ((((int)threadIdx.x) + 21) / 13)) + ry_outer_outer)) && ((((((((int)blockIdx.x) % 448) >> 4) * 8) + ((((int)threadIdx.x) + 21) / 13)) + ry_outer_outer) < 227)) && (3 <= ((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 13)))) && (((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 13)) < 227)) ? placeholder[((((((((((((int)threadIdx.x) + 112) / 91) * 50176) + (((((int)blockIdx.x) % 448) >> 4) * 1792)) + (((((int)threadIdx.x) + 21) / 13) * 224)) + (ry_outer_outer * 224)) + ((((int)blockIdx.x) & 15) * 14)) + rx_outer_outer) + ((((int)threadIdx.x) + 8) % 13)) - 675)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 168)] = (((((3 <= (((((((int)blockIdx.x) % 448) >> 4) * 8) + (((((int)threadIdx.x) + 77) % 91) / 13)) + ry_outer_outer)) && ((((((((int)blockIdx.x) % 448) >> 4) * 8) + (((((int)threadIdx.x) + 77) % 91) / 13)) + ry_outer_outer) < 227)) && (3 <= ((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 13)))) && (((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 13)) < 227)) ? placeholder[((((((((((((int)threadIdx.x) + 168) / 91) * 50176) + (((((int)blockIdx.x) % 448) >> 4) * 1792)) + ((((((int)threadIdx.x) + 77) % 91) / 13) * 224)) + (ry_outer_outer * 224)) + ((((int)blockIdx.x) & 15) * 14)) + rx_outer_outer) + ((((int)threadIdx.x) + 12) % 13)) - 675)] : 0.000000e+00f);
      if (((int)threadIdx.x) < 49) {
        pad_temp_shared[(((int)threadIdx.x) + 224)] = (((((((((((int)blockIdx.x) % 448) >> 4) * 8) + ((((int)threadIdx.x) + 42) / 13)) + ry_outer_outer) < 227) && (3 <= ((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 3) % 13)))) && (((((((int)blockIdx.x) & 15) * 14) + rx_outer_outer) + ((((int)threadIdx.x) + 3) % 13)) < 227)) ? placeholder[((((((((((((int)threadIdx.x) + 224) / 91) * 50176) + (((((int)blockIdx.x) % 448) >> 4) * 1792)) + (((((int)threadIdx.x) + 42) / 13) * 224)) + (ry_outer_outer * 224)) + ((((int)blockIdx.x) & 15) * 14)) + rx_outer_outer) + ((((int)threadIdx.x) + 3) % 13)) - 675)] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 48) {
        placeholder_shared[((int)threadIdx.x)] = placeholder1[(((((((int)blockIdx.x) / 448) * 2352) + (((int)threadIdx.x) * 49)) + (ry_outer_outer * 7)) + rx_outer_outer)];
      }
      __syncthreads();
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[((((int)threadIdx.x) / 28) * 24)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 1)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 2)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 3)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 4)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 5)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 6)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 7)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 8)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 9)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 10)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 11)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 12)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 13)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 14)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 15)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 16)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 17)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 18)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 19)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 20)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2))] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 21)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 91)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 22)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((((((int)threadIdx.x) % 28) / 7) * 26) + ((((int)threadIdx.x) % 7) * 2)) + 182)] * placeholder_shared[(((((int)threadIdx.x) / 28) * 24) + 23)]));
    }
  }
  for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
    T_relu[((((((((((int)blockIdx.x) / 448) * 200704) + ((((int)threadIdx.x) / 28) * 100352)) + (ax1_inner * 12544)) + (((((int)blockIdx.x) % 448) >> 4) * 448)) + (((((int)threadIdx.x) % 28) / 7) * 112)) + ((((int)blockIdx.x) & 15) * 7)) + (((int)threadIdx.x) % 7))] = max(((conv2d_nchw[ax1_inner] + placeholder2[((((((int)blockIdx.x) / 448) * 16) + ((((int)threadIdx.x) / 28) * 8)) + ax1_inner)]) + placeholder3[((((((int)blockIdx.x) / 448) * 16) + ((((int)threadIdx.x) / 28) * 8)) + ax1_inner)]), 0.000000e+00f);
  }

}
