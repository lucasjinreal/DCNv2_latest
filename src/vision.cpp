#include "dcn_v2.h"
#include <torch/script.h>

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  //m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
  //m.def("dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward, "dcn_v2_psroi_pooling_forward");
  //m.def("dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward, "dcn_v2_psroi_pooling_backward");
//}

static auto registry = torch::RegisterOperators("mynamespace::dcn_v2_forward", &dcn_v2_forward)
.op("mynamespace::dcn_v2_backward", &dcn_v2_backward)
.op("mynamespace::dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward)
.op("mynamespace::dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward);
