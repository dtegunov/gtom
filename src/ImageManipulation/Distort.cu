#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "DeviceFunctions.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"


namespace gtom
{
    __global__ void DistortKernel(cudaTex t_image, int2 dimsinput, tfloat* d_output, uint dimoutput, float2* d_offsets, glm::mat2* d_transforms);
    __global__ void WarpKernel(cudaTex t_image, tfloat* d_output, int2 dims, cudaTex t_warpx, cudaTex t_warpy);

    //////////////////////////
    //Corrects for beam tilt//
    //////////////////////////

    void d_DistortImages(tfloat* d_input, int2 dimsinput, tfloat* d_output, int2 dimsoutput, float2* h_offsets, float* h_rotations, float3* h_scales, uint batch)
    {
        cudaArray_t a_image;
        cudaTex t_image;

        {
            d_BindTextureTo3DArray(d_input, a_image, t_image, toInt3(dimsinput.x, dimsinput.y, batch), cudaFilterModePoint, false);
        }

        glm::mat2* h_transforms = (glm::mat2*)malloc(batch * sizeof(glm::mat2));
        for (int i = 0; i < batch; i++)
            h_transforms[i] = Matrix2Rotation(h_rotations[i]) * Matrix2Rotation(h_scales[i].z) * Matrix2Scale(tfloat2(1.0f / h_scales[i].x, 1.0f / h_scales[i].y)) * Matrix2Rotation(-h_scales[i].z);

        glm::mat2* d_transforms = (glm::mat2*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat2));
        free(h_transforms);

        float2* d_offsets = (float2*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float2));

        dim3 grid = dim3(tmin(32768, (Elements2(dimsoutput) + 127) / 128), batch, 1);
        DistortKernel << <grid, 128 >> > (t_image, dimsinput, d_output, dimsoutput.x, d_offsets, d_transforms);

        cudaFree(d_offsets);
        cudaFree(d_transforms);

        {
            cudaDestroyTextureObject(t_image);
            cudaFreeArray(a_image);
        }
    }

    void d_WarpImage(tfloat* d_input, tfloat* d_output, int2 dims, tfloat* h_warpx, tfloat* h_warpy, int2 dimswarp)
    {
        cudaArray_t a_image, a_warpx, a_warpy;
        cudaTex t_image, t_warpx, t_warpy;

        tfloat* d_warpx = (tfloat*)CudaMallocFromHostArray(h_warpx, Elements2(dimswarp) * sizeof(tfloat));
        tfloat* d_warpy = (tfloat*)CudaMallocFromHostArray(h_warpy, Elements2(dimswarp) * sizeof(tfloat));

        {
            d_BindTextureToArray(d_input, a_image, t_image, dims, cudaFilterModePoint, false);
            d_BindTextureToArray(d_warpx, a_warpx, t_warpx, dimswarp, cudaFilterModeLinear, true);
            d_BindTextureToArray(d_warpy, a_warpy, t_warpy, dimswarp, cudaFilterModeLinear, true);
        }

        dim3 grid = dim3(tmin(32768, (Elements2(dims) + 127) / 128), 1, 1);
        WarpKernel << <grid, 128 >> > (t_image, d_output, dims, t_warpx, t_warpy);

        cudaFree(d_warpx);
        cudaFree(d_warpy);

        cudaDestroyTextureObject(t_image);
        cudaFreeArray(a_image);

        cudaDestroyTextureObject(t_warpx);
        cudaFreeArray(a_warpx);
        cudaDestroyTextureObject(t_warpy);
        cudaFreeArray(a_warpy);
    }

    __global__ void DistortKernel(cudaTex t_image, int2 dimsinput, tfloat* d_output, uint dimoutput, float2* d_offsets, glm::mat2* d_transforms)
    {
        d_output += dimoutput * dimoutput * blockIdx.y;
        float zcoord = blockIdx.y + 0.5f;
        int2 inputcenter = make_int2(dimsinput.x / 2, dimsinput.y / 2);
        int outputcenter = dimoutput / 2;

        glm::mat2 transform = d_transforms[blockIdx.y];
        float2 offset = d_offsets[blockIdx.y];

        for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < dimoutput * dimoutput; id += gridDim.x * blockDim.x)
        {
            uint idx = id % dimoutput;
            uint idy = id / dimoutput;

            int posx = (int)idx - outputcenter;
            int posy = (int)idy - outputcenter;

            glm::vec2 pos = transform * glm::vec2(posx, posy);

            pos.x += inputcenter.x - offset.x;
            pos.y += inputcenter.y - offset.y;


            //tfloat val = tex3D<tfloat>(t_image, pos.x + 0.5f, pos.y + 0.5f, zcoord);
            tfloat val = 0;

            for (int y = -4; y <= 4; y++)
            {
                float yy = floor(pos.y) + y;
                float sincy = sinc(pos.y - yy);
                float yy2 = pos.y - yy;
                yy2 *= yy2;
                yy += 0.5f;

                for (int x = -4; x <= 4; x++)
                {
                    float xx = floor(pos.x) + x;
                    float sincx = sinc(pos.x - xx);
                    float xx2 = pos.x - xx;
                    xx2 *= xx2;
                    float r2 = xx2 + yy2;

                    if (r2 > 16)
                        continue;

                    float hanning = 1.0f + cos(PI * sqrt(r2) / 4);

                    val += tex3D<tfloat>(t_image, xx + 0.5f, yy, zcoord) * sincy * sincx * hanning;
                }
            }

            d_output[id] = val * 0.5f;
        }
    }

    __global__ void WarpKernel(cudaTex t_image, tfloat* d_output, int2 dims, cudaTex t_warpx, cudaTex t_warpy)
    {
        for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < Elements2(dims); id += gridDim.x * blockDim.x)
        {
            uint idx = id % (uint)dims.x;
            uint idy = id / (uint)dims.x;

            int posx = (int)idx;
            int posy = (int)idy;

            float2 warppos = make_float2((float)posx / (dims.x - 1), (float)posy / (dims.y - 1));
            float2 pos = make_float2(posx - tex2D<tfloat>(t_warpx, warppos.x, warppos.y),
                                     posy - tex2D<tfloat>(t_warpy, warppos.x, warppos.y));

            //tfloat val = tex3D<tfloat>(t_image, pos.x + 0.5f, pos.y + 0.5f, zcoord);
            tfloat val = 0;

            for (int y = -8; y <= 8; y++)
            {
                float yy = floor(pos.y) + y;
                float sincy = sinc(pos.y - yy);
                float yy2 = pos.y - yy;
                yy2 *= yy2;
                yy += 0.5f;

                for (int x = -8; x <= 8; x++)
                {
                    float xx = floor(pos.x) + x;
                    float sincx = sinc(pos.x - xx);
                    float xx2 = pos.x - xx;
                    xx2 *= xx2;
                    float r2 = xx2 + yy2;

                    if (r2 > 64)
                        continue;

                    float hanning = 1.0f + cos(PI * sqrt(r2) / 8);

                    val += tex2D<tfloat>(t_image, xx + 0.5f, yy) * sincy * sincx * hanning;
                }
            }

            d_output[id] = val * 0.5f;
        }
    }
}