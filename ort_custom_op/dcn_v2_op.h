#pragma once
#include <iostream>
#include "onnxruntime_cxx_api.h"

struct Input {
  const char* name;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

template <typename T>
struct DCNv2Kernel {
	private:
   Ort::CustomOpApi ort_;

	public:
  DCNv2Kernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {}

  void Compute(OrtKernelContext* context);
};


struct DCNv2CustomOp: Ort::CustomOpBase<DCNv2CustomOp, DCNv2Kernel<float>> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new DCNv2Kernel<float>(api, info); };
  const char* GetName() const { return "testdcn"; };

  size_t GetInputTypeCount() const { return 14; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

template <typename T>
void print_vector(const std::vector<T> & vec)
{
  for (int i = 0; i < vec.size(); i++)
    std::cout << vec[i] << ", ";
}

#include "dcn_v2_op.cc"
