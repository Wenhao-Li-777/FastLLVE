#include <torch/extension.h>

/* CUDA Forward Declarations */

void IALUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output);


void IALUTTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut);


void IALUT_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    IALUTTransformForwardCUDAKernelLauncher(input, lut, output);
}


void IALUT_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    IALUTTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, grad_inp, grad_lut);
}


void IALUT_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output);


void IALUT_transform_cpu_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut);


/* C++ Interfaces */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void IALUT_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(output);

        IALUT_transform_cuda_forward(input, lut, output);
    } else {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(output);

        IALUT_transform_cpu_forward(input, lut, output);
    }
}


void IALUT_transform_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);

        IALUT_transform_cuda_backward(grad_output, input, lut, grad_inp, grad_lut);
    } else {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);

        IALUT_transform_cpu_backward(grad_output, input, lut, grad_inp, grad_lut);
    }
}


/* Interfaces Binding */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("IALUT_cforward", &IALUT_transform_forward, "IALUT-Transform forward");
  m.def("IALUT_cbackward", &IALUT_transform_backward, "IALUT-Transform backward");
}

