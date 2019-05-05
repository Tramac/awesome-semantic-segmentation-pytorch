#include "ca.h"
#include "psa.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ca_forward", &ca_forward, "ca_forward");
  m.def("ca_backward", &ca_backward, "ca_backward");
  m.def("ca_map_forward", &ca_map_forward, "ca_map_forward");
  m.def("ca_map_backward", &ca_map_backward, "ca_map_backward");
  m.def("psa_forward", &psa_forward, "psa_forward");
  m.def("psa_backward", &psa_backward, "psa_backward");
}