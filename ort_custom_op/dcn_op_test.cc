#include <iostream>
#include "dcn_v2_op.h"
#include "onnxruntime_cxx_api.h"

typedef const char* PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE MODEL_URI = TSTR("../../model.onnx");

template <typename T>
bool TestInference(Ort::Env& env, T model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   OrtCustomOpDomain* custom_op_domain_ptr) {
  Ort::SessionOptions session_options;
  std::cout << "Running simple inference with default provider" << std::endl;

  if (custom_op_domain_ptr) {
    session_options.Add(custom_op_domain_ptr);
  }

  Ort::Session session(env, model_uri, session_options);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> input_names;

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::cout << "run model" << std::endl;
  std::vector<Ort::Value> ort_outputs;
  ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);

  // print result
  Ort::Value* out_tensor = &ort_outputs[0];
  auto type_info = out_tensor->GetTensorTypeAndShapeInfo();
  auto out_shape = type_info.GetShape();
  size_t total_len = type_info.GetElementCount();
  std::cout << "output shape: ";
  print_vector(out_shape); std::cout << std::endl;
  float* out_ptr = out_tensor->GetTensorMutableData<float>();
  for (int i = 0; i < total_len; i++)
  {
      if (i % (out_shape[2]*out_shape[3]) == 0)
          std::cout << std::endl;
      std::cout << out_ptr[i] << ' ';
  }
  std::cout << std::endl;

  std::cout << "end" << std::endl;
  return true;
}

int main(int argc, char** argv) {

  Ort::Env env_= Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");

  std::vector<Input> inputs(3);
  auto input = inputs.begin();
  input->name = "input";
  input->dims = {1, 1, 4, 4};
  std::vector<float> input_v(16);
  for (int i = 0; i < input_v.size(); i++)
      input_v[i] = i;
  //input->values = std::vector<float>(16, 1.0f);
  input->values = input_v;

  input = std::next(input, 1);
  input->name = "offset";
  input->dims = {1, 18, 4, 4};
  //input->values = std::vector<float>(288, 1.0f);
  std::vector<float> offset_v(288);
  for (int i = 0; i < offset_v.size(); i++)
      offset_v[i] = i / 50.f;
  input->values = offset_v;

  input = std::next(input, 1);
  input->name = "mask";
  input->dims = {1, 9, 4, 4};
  input->values = std::vector<float>(144, 1.0f);

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2, 1, 2};
  std::vector<float> expected_values_y = { 3.0000f, -1.0000f, -1.0000f,  1.0000f, 2.9996f, -0.9996f, -0.9999f,  0.9999f,  -0.9996f,  2.9996f, -1.0000f,  1.0000f};

  DCNv2CustomOp custom_op;
  Ort::CustomOpDomain custom_op_domain("mydomain");
  custom_op_domain.Add(&custom_op);

  return TestInference(env_, MODEL_URI, inputs, "output", expected_dims_y, expected_values_y, custom_op_domain);
}
