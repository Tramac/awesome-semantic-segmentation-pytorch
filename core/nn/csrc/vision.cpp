#include "ca.h"
#include "psa.h"
#include "syncbn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ca_forward", &ca_forward, "ca_forward");
  m.def("ca_backward", &ca_backward, "ca_backward");
  m.def("ca_map_forward", &ca_map_forward, "ca_map_forward");
  m.def("ca_map_backward", &ca_map_backward, "ca_map_backward");
  m.def("psa_forward", &psa_forward, "psa_forward");
  m.def("psa_backward", &psa_backward, "psa_backward");
  m.def("batchnorm_forward", &batchnorm_forward, "batchnorm_forward");
  m.def("inp_batchnorm_forward", &inp_batchnorm_forward, "inp_batchnorm_forward");
  m.def("batchnorm_backward", &batchnorm_backward, "batchnorm_backward");
  m.def("inp_batchnorm_backward", &inp_batchnorm_backward, "inp_batchnorm_backward");
  m.def("expectation_forward", &expectation_forward, "expectation_forward");
  m.def("expectation_backward", &expectation_backward, "expectation_backward");
  m.def("inp_expectation_backward", &inp_expectation_backward, "inp_expectation_backward");
}