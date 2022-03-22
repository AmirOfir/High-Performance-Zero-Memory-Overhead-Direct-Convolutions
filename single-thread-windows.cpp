#pragma once

#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace torch;
using namespace std;

#ifndef min_
#define min_(a,b)            (((a) > (b)) ? (b) : (a))
#define max_(a,b)            (((a) > (b)) ? (a) : (b))
#endif

inline void MulAddArray(float* target, const float* src, float scalar, int64_t length)
{
    for (int64_t l = 0; l < length; ++l)
    {
        target[l] += src[l] * scalar;
    }
}

// ----------------- Convolution implementation -----------------
/*
    Implementation of "High Performance Zero-Memory Overhead Direct Convolutions"
    Algorithm 2 (single threaded)

    Input shape (batch, height, width, channels)
    Weights shape (height, width, in channel, out channel)
    Output shape (batch, height, width, channels)
    
    Limitation: Stride=1, Pad=0
*/

Tensor HWC_DirectConvolution_Stride1_SingleThread_NoSimd(const Tensor& input, const Tensor& weights)
{
    const int batch_size = input.size(0);

    const int input_height = input.size(1);
    const int input_width = input.size(2);
    const int in_channels = input.size(3);

    const int kernel_height = weights.size(0);
    const int kernel_width = weights.size(1);
    const int out_channels = weights.size(3);
    
    const int width_out = (input_width - kernel_width) + 1;
    const int height_out = (input_height - kernel_height) + 1;

    const int height_size = input_width * in_channels;

    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    Tensor result = torch::zeros({ batch_size, height_out, width_out, out_channels }, options).contiguous();

    const float *weightsPtr = weights.data_ptr<float>();
    for (int batch_curr = 0; batch_curr < batch_size; ++batch_curr)
    {
        float* outPtr = result[batch_curr].data_ptr<float>();
        const float* inPtr = input[batch_curr].data_ptr<float>();
        
        int in_height;
        int out_height = 0;
        for (int y = 0; y < height_out; y++)
        {
            in_height = out_height;
            for (int kernel_y = 0; kernel_y < kernel_height; kernel_y++)
            {
                int in_width_channel = 0;
                for (int kernel_x = 0; kernel_x < kernel_width; kernel_x++)
                {
                    for (int in_channel = 0; in_channel < in_channels; in_channel++)
                    {
                        const int weight_pixel = (out_channels *
                            (in_channel +
                                (in_channels *
                                    (kernel_x + kernel_y * kernel_width)
                                    )
                                )
                        );
                        const float* weight_pixel_ptr = weightsPtr + weight_pixel;

                        int in_width = in_channels + in_width_channel;
                        int out_width = 0;
                        for (int x = 0; x < width_out; x++)
                        {
                            const float input_value = inPtr[in_height + in_width + in_channel];
                            float* out_pixel_ptr = outPtr + out_width;

                            MulAddArray(out_pixel_ptr, weight_pixel_ptr, input_value, out_channels);
                            //for (int out_channel = 0; out_channel < out_channels; out_channel++)
                            //{
                            //    out_pixel_ptr[out_channel] += input_value * weight_pixel_ptr[out_channel];

                            //} // out_channels

                            out_width += out_channels;
                            in_width += in_channels;
                        }
                    } // in_channels

                    in_width_channel += in_channels;
                } // kernel_width

                in_height += height_size;
            } // kernel_height

            out_height += height_size;
            outPtr += width_out * out_channels;
        }
    }
    return result;
}

/*
* My implementation for CHW Direct convolution
* Single threaded
* Input of shape (batch, channels, height, width)
* Weights of shape (height, width, in channel, out channel)
* Stride=1
* Pad=0
* NO SIMD
*/
void CHW_DirectConvolution_Stride1_SingleThread(const Tensor& input, Tensor &output, const Tensor& weights)
{
    const int batch_size = input.size(0);
    const int C_i = input.size(1);
    const int H_i = input.size(2);
    const int W_i = input.size(3);
    const int H_f = weights.size(0);
    const int W_f = weights.size(1);
    const int C_o = output.size(1);
    const int W_o = output.size(2);
    const int H_o = output.size(3);

    const int output_plane_size = W_o * H_o;
    int eol_jump = W_o - W_i;

    for (int batch_curr = 0; batch_curr < batch_size; ++batch_curr)
    {
        float* outPtr = output[batch_curr].data_ptr<float>();
        float* inPtr = input[batch_curr].data_ptr<float>();
        float* weightPtr = weights.data_ptr<float>();
        for (int in_channel = 0; in_channel < C_i; ++in_channel)
        {
            for (int kernel_row_offset = 0; kernel_row_offset < H_f; ++kernel_row_offset)
            {
                float* inStart = inPtr + (kernel_row_offset * W_o);
                for (int kernel_col_offset = 0; kernel_col_offset < W_f; ++kernel_col_offset)
                {
                    float* outCurr = outPtr;

                    for (int out_channel = 0; out_channel < C_o; ++out_channel)
                    {
                        float* inCurr = inStart;

                        for (int h_o = 0; h_o < H_o; ++h_o)
                        {
                            for (int w_o = 0; w_o < W_o; ++w_o)
                            {
                                *outPtr += (*inCurr) * (*weightPtr);
                                ++outCurr;
                                ++inCurr;
                            }
                            inCurr += eol_jump;

                        }

                        ++weightPtr;
                    }

                    ++inStart;
                }
            }
            inPtr += H_i * W_i;
        }
    }
}

/*
* My implementation for CHW Direct convolution 
* Single threaded
* Input of shape (batch, channels, height, width)
* Weights of shape (height, width, in channel, out channel)
* Stride=1
* Pad=0
* NO SIMD
*/
Tensor CHW_DirectConvolution_Stride1_SingleThread(const Tensor& input, const Tensor& weights)
{
    const int batch_size = input.size(0);
    const int C_i = input.size(1);
    const int H_i = input.size(2);
    const int W_i = input.size(3);
    const int H_f = weights.size(0);
    const int W_f = weights.size(1);
    const int C_o = weights.size(3);
    const int W_o = (W_i - W_f) + 1;
    const int H_o = (H_i - H_f) + 1;

    const int output_plane_size = W_o * H_o;
    int eol_jump = W_o - W_i;
    
    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    Tensor result = torch::zeros({ batch_size, C_o, H_o, W_o }, options).contiguous();
    
    CHW_DirectConvolution_Stride1_SingleThread(input, result, weights);
    
    return result;
}

Tensor CHW_DirectConvolution_Stride1_Pad(const Tensor& input, const Tensor& weights, int pad)
{
    const int batch_size = input.size(0);
    const int C_i = input.size(1);
    const int H_i = input.size(2);
    const int W_i = input.size(3);
    const int H_f = weights.size(0);
    const int W_f = weights.size(1);
    const int C_o = weights.size(3);
    const int W_o = (W_i - W_f + pad + pad) + 1;
    const int H_o = (H_i - H_f + pad + pad) + 1;

    const int output_plane_size = W_o * H_o;
    int eol_jump = W_o - W_i;

    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    Tensor result = torch::zeros({ batch_size, C_o, H_o, W_o }, options).contiguous();


    for (int batch_curr = 0; batch_curr < batch_size; ++batch_curr)
    {
        float* outPtr = result[batch_curr].data_ptr<float>();
        float* inPtr = input[batch_curr].data_ptr<float>();
        float* weightPtr = weights.data_ptr<float>();
        for (int in_channel = 0; in_channel < C_i; ++in_channel)
        {
            for (int kernel_row_offset = 0; kernel_row_offset < H_f; ++kernel_row_offset)
            {
                float* inStart = inPtr + (kernel_row_offset * W_o);
                for (int kernel_col_offset = 0; kernel_col_offset < W_f; ++kernel_col_offset)
                {
                    float* outCurr = outPtr;

                    for (int out_channel = 0; out_channel < C_o; ++out_channel)
                    {
                        float* inCurr = inStart;

                        for (int p = 0; p < pad; ++p)
                            outPtr += W_o;
                        for (int h_o = 0; h_o < H_o; ++h_o)
                        {
                            outPtr += pad;
                            for (int w_o = pad; w_o < W_o - pad; ++w_o)
                            {
                                *outPtr += (*inCurr) * (*weightPtr);
                                ++outCurr;
                                ++inCurr;
                            }
                            inCurr += eol_jump;
                            outPtr += pad;
                        }
                        for (int p = 0; p < pad; ++p)
                            outPtr += W_o;

                        ++weightPtr;
                    }

                    ++inStart;
                }
            }
            inPtr += H_i * W_i;
        }
    }
    return result;
}

Tensor CHW_DirectConvolution_Stride1_SIMD(const Tensor& input, const Tensor& weights)
{
    const int batch_size = input.size(0);
    const int C_i = input.size(1);
    const int H_i = input.size(2);
    const int W_i = input.size(3);
    const int H_f = weights.size(0);
    const int W_f = weights.size(1);
    const int C_o = weights.size(3);
    const int W_o = (W_i - W_f) + 1;
    const int H_o = (H_i - H_f) + 1;

    const int output_plane_size = W_o * H_o;
    int eol_jump = W_o - W_i;

    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    Tensor result = torch::zeros({ batch_size, C_o, H_o, W_o }, options).contiguous();

    for (int batch_curr = 0; batch_curr < batch_size; ++batch_curr)
    {
        float* outPtr = result[batch_curr].data_ptr<float>();
        float* inPtr = input[batch_curr].data_ptr<float>();
        float* weightPtr = weights.data_ptr<float>();
        for (int in_channel = 0; in_channel < C_i; ++in_channel)
        {
            for (int kernel_row_offset = 0; kernel_row_offset < H_f; ++kernel_row_offset)
            {
                for (int kernel_col_offset = 0; kernel_col_offset < W_f; ++kernel_col_offset)
                {
                    float* inStart = inPtr + (kernel_row_offset * W_o) + kernel_col_offset;
                    float* outCurr = outPtr;

                    for (int out_channel = 0; out_channel < C_o; ++out_channel)
                    {
                        float* inCurr = inStart;

                        for (int h_o = 0; h_o < H_o; ++h_o)
                        {
                            for (int w_o = 0; w_o < W_o; ++w_o)
                            {
                                *outPtr += (*inCurr) * (*weightPtr);
                                ++outCurr;
                                ++inCurr;
                            }
                            inCurr += eol_jump;
                        }

                        ++weightPtr;
                    }

                }
            }
            inPtr += H_i * W_i;
        }
    }
    return result;
}
