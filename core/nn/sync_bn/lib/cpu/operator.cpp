#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batchnorm_forward", &BatchNorm_Forward_CPU, "BatchNorm forward (CPU)");
    m.def("batchnorm_backward", &BatchNorm_Backward_CPU, "BatchNorm backward (CPU)");
    m.def("sumsquare_forward", &Sum_Square_Forward_CPU, "SumSqu forward (CPU)");
    m.def("sumsquare_backward", &Sum_Square_Backward_CPU, "SumSqu backward (CPU)");
}