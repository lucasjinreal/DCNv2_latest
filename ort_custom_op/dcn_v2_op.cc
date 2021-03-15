#include <iostream>
#include "Eigen/Dense"
#include "onnxruntime_cxx_api.h"
#include "dcn_v2_im2col_cpu.h"

template <typename T>
using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixXfR;
using EigenMatrixMap = Eigen::Map<MatrixXfR>;
using ConstEigenMatrixMap = Eigen::Map<const MatrixXfR>;

template <typename T>
void DCNv2Kernel<T>::Compute(OrtKernelContext* context) {
  // Setup inputs tensors
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const T* X_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_X));
  const OrtValue* input_weight = ort_.KernelContext_GetInput(context, 1);
  const T* weight_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_weight));
  const OrtValue* input_bias = ort_.KernelContext_GetInput(context, 2);
  const T* bias_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_bias));
  const OrtValue* input_offset = ort_.KernelContext_GetInput(context, 3);
  const T* offset_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_offset));
  const OrtValue* input_mask = ort_.KernelContext_GetInput(context, 4);
  const T* mask_data = reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_mask));

  // Setup inputs values
  const OrtValue* input_kernel_h = ort_.KernelContext_GetInput(context, 5);
  const int64_t kernel_h = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_kernel_h)))[0]);
  const OrtValue* input_kernel_w = ort_.KernelContext_GetInput(context, 6);
  const int64_t kernel_w = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_kernel_w)))[0]);
  const OrtValue* input_stride_h = ort_.KernelContext_GetInput(context, 7);
  const int64_t stride_h = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_stride_h)))[0]);
  const OrtValue* input_stride_w = ort_.KernelContext_GetInput(context, 8);
  const int64_t stride_w = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_stride_w)))[0]);
  const OrtValue* input_padding_h = ort_.KernelContext_GetInput(context, 9);
  const int64_t padding_h = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_padding_h)))[0]);
  const OrtValue* input_padding_w = ort_.KernelContext_GetInput(context, 10);
  const int64_t padding_w = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_padding_w)))[0]);
  const OrtValue* input_dilation_h = ort_.KernelContext_GetInput(context, 11);
  const int64_t dilation_h = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_dilation_h)))[0]);
  const OrtValue* input_dilation_w = ort_.KernelContext_GetInput(context, 12);
  const int64_t dilation_w = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_dilation_w)))[0]);
  const OrtValue* input_deformable_groups = ort_.KernelContext_GetInput(context, 13);
  const int64_t deformable_groups = int64_t((reinterpret_cast<const T*>(ort_.GetTensorData<const T*>(input_deformable_groups)))[0]);
  
  // get input dims
  OrtTensorDimensions input_dim(ort_, input_X);
  const int64_t batch = input_dim[0];
  const int64_t channels = input_dim[1];
  const int64_t height = input_dim[2];
  const int64_t width = input_dim[3];
  const int64_t input_sample = channels*height*width;

  OrtTensorDimensions weight_dim(ort_, input_weight);
  const int64_t channels_out = weight_dim[0];
  const int64_t channels_kernel = weight_dim[1];
  const int64_t kernel_h_ = weight_dim[2];
  const int64_t kernel_w_ = weight_dim[3];

  OrtTensorDimensions offset_dim(ort_, input_offset);
  const int64_t offset_sample = offset_dim[1]*offset_dim[2]*offset_dim[3];

  OrtTensorDimensions mask_dim(ort_, input_mask);
  const int64_t mask_sample = mask_dim[1]*mask_dim[2]*mask_dim[3];

  assert(kernel_h_ == kernel_h && kernel_w_ == kernel_w);
               //"Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);
  assert(channels == channels_kernel);
               //"Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

  // for debug
  //std::cout << "input_dim : ";
  //print_vector(input_dim); std::cout << std::endl;
  //std::cout << "weight_dim : ";
  //print_vector(weight_dim); std::cout << std::endl;
  //std::cout << "offset_dim : ";
  //print_vector(offset_dim); std::cout << std::endl;
  //std::cout << "mask_dim : ";
  //print_vector(mask_dim); std::cout << std::endl;
  
  // out dim
  const int64_t height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int64_t width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  std::vector<int64_t> out_dim = {batch, channels_out, height_out, width_out};
  const int64_t out_sample = channels_out*height_out*width_out;

  // Setup output
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, out_dim.data(), out_dim.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // set others
  MatrixXfR ones = MatrixXfR::Ones(height_out, width_out);
  MatrixXfR columns = MatrixXfR::Zero(channels * kernel_h * kernel_w, height_out * width_out);
  ConstEigenMatrixMap weight(weight_data, channels_out, channels * kernel_h * kernel_w);

  // bias
  int64_t n_ = height_out * width_out;
  int64_t m_ = channels_out;
  int64_t k_ = 1;
  ones.resize(k_, n_);
  ConstEigenMatrixMap bias(bias_data, m_, k_);
  MatrixXfR ones_T = bias * ones;
  ones_T.resize(1, out_sample);

  // Do computation
  using scalar_t = float;
  for (int b = 0; b < batch; b++)
  {   
      EigenMatrixMap output_n(out, 1, out_sample);
      output_n += ones_T;

      modulated_deformable_im2col_cpu(X_data,
                                       offset_data,
                                       mask_data,
                                       1, channels, height, width,
                                       height_out, width_out, kernel_h, kernel_w,
                                       padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                                       deformable_groups,
                                       (scalar_t*)columns.data());

      //(k * m)  x  (m * n)
      // Y = WC
      //std::cout << "channels : " << channels << std::endl;
      //std::cout << "height: " << height<< std::endl;
      //std::cout << "width: " << width<< std::endl;
      //std::cout << "height_out: " << height_out<< std::endl;
      //std::cout << "width_out: " << width_out<< std::endl;

      //std::cout << "kernel_h: " << kernel_h<< std::endl;
      //std::cout << "kernel_w: " << kernel_w<< std::endl;
      //std::cout << "padding_h: " << padding_h<< std::endl;
      //std::cout << "padding_w: " << padding_w<< std::endl;

      //std::cout << "stride_h: " << stride_h<< std::endl;
      //std::cout << "stride_w: " << stride_w<< std::endl;
      //std::cout << "dilation_h: " << dilation_h<< std::endl;
      //std::cout << "dilation_w: " << dilation_w<< std::endl;
      //std::cout << "deformable_groups: " << deformable_groups<< std::endl;

      //std::cout << "input data" << std::endl;
      //for (int i = 0; i < input_sample; i++)
          //std::cout << X_data[i] << ' ';
      //std::cout << std::endl;

      //std::cout << "offset data" << std::endl;
      //for (int i = 0; i < offset_sample; i++)
          //std::cout << offset_data[i] << ' ';
      //std::cout << std::endl;

      //std::cout << "mask data" << std::endl;
      //for (int i = 0; i < mask_sample; i++)
          //std::cout << mask_data[i] << ' ';
      //std::cout << std::endl;

      //std::cout << "columns" << std::endl;
      //std::cout << columns << std::endl;

      // eigen implementation    
      MatrixXfR product = weight * columns;
      product.resize(1, out_sample);
      output_n += product;

      // update ptr
      X_data += input_sample;
      offset_data += offset_sample;
      mask_data += mask_sample;
      out += out_sample;
  }
}
