#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor
dcn_v2_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const at::Tensor &mask,
               const at::Tensor kernel_h,
               const at::Tensor kernel_w,
               const at::Tensor stride_h,
               const at::Tensor stride_w,
               const at::Tensor pad_h,
               const at::Tensor pad_w,
               const at::Tensor dilation_h,
               const at::Tensor dilation_w,
               const at::Tensor deformable_group)
{
    int _kernel_h = int(kernel_h.data<float>()[0]);
    int _kernel_w = int(kernel_w.data<float>()[0]);
    int _stride_h = int(stride_h.data<float>()[0]);
    int _stride_w = int(stride_w.data<float>()[0]);
    int _pad_h = int(pad_h.data<float>()[0]);
    int _pad_w = int(pad_w.data<float>()[0]);
    int _dilation_h = int(dilation_h.data<float>()[0]);
    int _dilation_w = int(dilation_w.data<float>()[0]);
    int _deformable_group = int(deformable_group.data<float>()[0]);

    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_cuda_forward(input, weight, bias, offset, mask,
                                   _kernel_h, _kernel_w,
                                   _stride_h, _stride_w,
                                   _pad_h, _pad_w,
                                   _dilation_h, _dilation_w,
                                   _deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_cpu_forward(input, weight, bias, offset, mask,
                                   _kernel_h, _kernel_w,
                                   _stride_h, _stride_w,
                                   _pad_h, _pad_w,
                                   _dilation_h, _dilation_w,
                                   _deformable_group);
    }
}

std::vector<at::Tensor>
dcn_v2_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &grad_output,
                const at::Tensor kernel_h, const at::Tensor kernel_w,
                const at::Tensor stride_h, const at::Tensor stride_w,
                const at::Tensor pad_h, const at::Tensor pad_w,
                const at::Tensor dilation_h, const at::Tensor dilation_w,
                const at::Tensor deformable_group)
{
    int _kernel_h = int(kernel_h.data<float>()[0]);
    int _kernel_w = int(kernel_w.data<float>()[0]);
    int _stride_h = int(stride_h.data<float>()[0]);
    int _stride_w = int(stride_w.data<float>()[0]);
    int _pad_h = int(pad_h.data<float>()[0]);
    int _pad_w = int(pad_w.data<float>()[0]);
    int _dilation_h = int(dilation_h.data<float>()[0]);
    int _dilation_w = int(dilation_w.data<float>()[0]);
    int _deformable_group = int(deformable_group.data<float>()[0]);

    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_cuda_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    mask,
                                    grad_output,
                                    _kernel_h, _kernel_w,
                                    _stride_h, _stride_w,
                                    _pad_h, _pad_w,
                                    _dilation_h, _dilation_w,
                                    _deformable_group);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_cpu_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    mask,
                                    grad_output,
                                    _kernel_h, _kernel_w,
                                    _stride_h, _stride_w,
                                    _pad_h, _pad_w,
                                    _dilation_h, _dilation_w,
                                    _deformable_group);
    }
}

std::tuple<at::Tensor, at::Tensor>
dcn_v2_psroi_pooling_forward(const at::Tensor &input,
                             const at::Tensor &bbox,
                             const at::Tensor &trans,
                             const at::Tensor &no_trans,
                             const at::Tensor &spatial_scale,
                             const at::Tensor &output_dim,
                             const at::Tensor &group_size,
                             const at::Tensor &pooled_size,
                             const at::Tensor &part_size,
                             const at::Tensor &sample_per_part,
                             const at::Tensor &trans_std)
{
    int _no_trans = int(no_trans.data<float>()[0]);
    float _spatial_scale = spatial_scale.data<float>()[0];
    int _output_dim = int(output_dim.data<float>()[0]);
    int _group_size = int(group_size.data<float>()[0]);
    int _pooled_size = int(pooled_size.data<float>()[0]);
    int _part_size = int(part_size.data<float>()[0]);
    int _sample_per_part = int(sample_per_part.data<float>()[0]);
    float _trans_std = trans_std.data<float>()[0];

    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_psroi_pooling_cuda_forward(input,
                                                 bbox,
                                                 trans,
                                                 _no_trans,
                                                 _spatial_scale,
                                                 _output_dim,
                                                 _group_size,
                                                 _pooled_size,
                                                 _part_size,
                                                 _sample_per_part,
                                                 _trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_psroi_pooling_cpu_forward(input,
                                                 bbox,
                                                 trans,
                                                 _no_trans,
                                                 _spatial_scale,
                                                 _output_dim,
                                                 _group_size,
                                                 _pooled_size,
                                                 _part_size,
                                                 _sample_per_part,
                                                 _trans_std);
    }
}

std::tuple<at::Tensor, at::Tensor>
dcn_v2_psroi_pooling_backward(const at::Tensor &out_grad,
                              const at::Tensor &input,
                              const at::Tensor &bbox,
                              const at::Tensor &trans,
                              const at::Tensor &top_count,
                              const at::Tensor &no_trans,
                              const at::Tensor &spatial_scale,
                              const at::Tensor &output_dim,
                              const at::Tensor &group_size,
                              const at::Tensor &pooled_size,
                              const at::Tensor &part_size,
                              const at::Tensor &sample_per_part,
                              const at::Tensor &trans_std)
{
    int _no_trans = int(no_trans.data<float>()[0]);
    float _spatial_scale = spatial_scale.data<float>()[0];
    int _output_dim = int(output_dim.data<float>()[0]);
    int _group_size = int(group_size.data<float>()[0]);
    int _pooled_size = int(pooled_size.data<float>()[0]);
    int _part_size = int(part_size.data<float>()[0]);
    int _sample_per_part = int(sample_per_part.data<float>()[0]);
    float _trans_std = trans_std.data<float>()[0];

    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dcn_v2_psroi_pooling_cuda_backward(out_grad,
                                                  input,
                                                  bbox,
                                                  trans,
                                                  top_count,
                                                  _no_trans,
                                                  _spatial_scale,
                                                  _output_dim,
                                                  _group_size,
                                                  _pooled_size,
                                                  _part_size,
                                                  _sample_per_part,
                                                  _trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else{
        return dcn_v2_psroi_pooling_cpu_backward(out_grad,
                                                  input,
                                                  bbox,
                                                  trans,
                                                  top_count,
                                                  _no_trans,
                                                  _spatial_scale,
                                                  _output_dim,
                                                  _group_size,
                                                  _pooled_size,
                                                  _part_size,
                                                  _sample_per_part,
                                                  _trans_std);
    }
}
