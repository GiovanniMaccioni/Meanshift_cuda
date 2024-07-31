#include <cuda.h>
#include<cuda_runtime.h>
#include "../include/defines.h"


__device__ __host__
inline float euclidian_norm2D(float x, float y)
{
    return sqrt(x*x + y*y);
}

__device__ __host__
inline bool is_neighbour2D(float c_x, float c_y, float p_x, float p_y, float bandwidth)
{
    return euclidian_norm2D(c_x - p_x, c_y - p_y) < bandwidth? true:false;
}

__global__
void centroid_convergence_kernel(float* d_centroid_x, float* d_centroid_y, float* d_points_x, float* d_points_y, int num_points)
{
    int thread_id = blockDim.x*blockIdx.x + threadIdx.x;

    if(thread_id < num_points)
    {   
        float meanshift_x = 0;
        float meanshift_y = 0;
        int num_neighbours = 0;

        for(int i = 0; i < num_points; i++)
        {

            if(is_neighbour2D(d_centroid_x[thread_id], d_centroid_y[thread_id], d_points_x[i], d_points_y[i], BANDWIDTH))
            {
                meanshift_x += d_points_x[i];
                meanshift_y += d_points_y[i];
                num_neighbours++;
            }
        }

        if(num_neighbours != 0)
        {
            d_centroid_x[thread_id] = meanshift_x / num_neighbours;
            d_centroid_y[thread_id] = meanshift_y / num_neighbours;
        }
    }
}


void meanshift_convergence(dataset2D centroids, dataset2D points, int num_points)
{
    float *d_centroid_x;
    float *d_centroid_y;

    float *d_points_x;
    float *d_points_y;

    cudaMalloc((void**) &d_centroid_x, sizeof(float)*num_points);
    cudaMemcpy(d_centroid_x, centroids.x, sizeof(float)*num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_centroid_y, sizeof(float)*num_points);
    cudaMemcpy(d_centroid_y, centroids.y, sizeof(float)*num_points, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_points_x, sizeof(float)*num_points);
    cudaMemcpy(d_points_x, points.x, sizeof(float)*num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_points_y, sizeof(float)*num_points);
    cudaMemcpy(d_points_y, points.y, sizeof(float)*num_points, cudaMemcpyHostToDevice);

    //KERNEL CALL
    //We will have a 1d block
    for(int i = 0; i < MAX_ITERS; i++)
    {
        centroid_convergence_kernel<<<ceil(num_points/(float)BLOCK_DIM), BLOCK_DIM>>>(d_centroid_x, d_centroid_y, d_points_x, d_points_y, num_points);
    }
    

    cudaMemcpy(centroids.x, d_centroid_x, sizeof(float)*num_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.y, d_centroid_y, sizeof(float)*num_points, cudaMemcpyDeviceToHost);

    cudaFree(d_centroid_x);
    cudaFree(d_centroid_y);
    cudaFree(d_points_x);
    cudaFree(d_points_y);

    return;
}

__global__
void merge_clusters_kernel(float* d_centroid_xi, float* d_centroid_yi, float* d_centroid_x, float* d_centroid_y, int* d_centroids_labels, int num_points, int label)
{
    int thread_id = blockDim.x*blockIdx.x + threadIdx.x;
    if (thread_id < num_points)
    {
        if(d_centroids_labels[thread_id] == -1)
            if (is_neighbour2D(*d_centroid_xi, *d_centroid_yi, d_centroid_x[thread_id], d_centroid_y[thread_id], BANDWIDTH))
                int old_label = atomicCAS(&d_centroids_labels[thread_id], -1, label);
    }
}

void meanshift_merge(dataset2D centroids, int num_points)
{
    float* d_centroid_x, *d_centroid_y;
    int* d_centroids_labels;

    int label = 0;
    //int* d_label;

    cudaMalloc((void**) &d_centroid_x, sizeof(float)*num_points);
    cudaMemcpy(d_centroid_x, centroids.x, sizeof(float)*num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_centroid_y, sizeof(float)*num_points);
    cudaMemcpy(d_centroid_y, centroids.y, sizeof(float)*num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_centroids_labels, sizeof(int)*num_points);
    cudaMemcpy(d_centroids_labels, centroids.labels, sizeof(int)*num_points, cudaMemcpyHostToDevice);


    //KERNEL CALL
    //We will have a 1d block
    for(int i = 0; i < num_points; i++)
    {
        merge_clusters_kernel<<<ceil(num_points/(float)BLOCK_DIM), BLOCK_DIM>>>(&d_centroid_x[i], &d_centroid_y[i], d_centroid_x, d_centroid_y, d_centroids_labels, num_points, label);
        cudaDeviceSynchronize();
        label++;
    }

    cudaMemcpy(centroids.labels, d_centroids_labels, sizeof(int)*num_points, cudaMemcpyDeviceToHost);

    cudaFree(d_centroid_x);
    cudaFree(d_centroid_y);
    cudaFree(d_centroids_labels);

    return;
}


//---------------------------------------------------------------------------------------------------


__global__
void centroid_convergence_kernel_2(float* d_centroid_x, float* d_centroid_y, float* d_points_x, float* d_points_y, int num_points, float* ms_x, float* ms_y, int* num_neighbours)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points)
    {
        if (is_neighbour2D(*d_centroid_x, *d_centroid_y, d_points_x[i], d_points_y[i], BANDWIDTH))
        {
            atomicAdd(ms_x, d_points_x[i]);
            atomicAdd(ms_y, d_points_y[i]);
            atomicAdd(num_neighbours, 1);
        }
    }
}

void meanshift_convergence_2(dataset2D centroids, dataset2D points, int num_points)
{
    float *d_centroid_x, *d_centroid_y;
    float *d_points_x, *d_points_y;
    
    float *d_ms_x, *d_ms_y;
    int *d_num_neighbours;

    cudaMalloc((void**)&d_centroid_x, sizeof(float) * num_points);
    cudaMemcpy(d_centroid_x, centroids.x, sizeof(float) * num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_centroid_y, sizeof(float) * num_points);
    cudaMemcpy(d_centroid_y, centroids.y, sizeof(float) * num_points, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_points_x, sizeof(float) * num_points);
    cudaMemcpy(d_points_x, points.x, sizeof(float) * num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_points_y, sizeof(float) * num_points);
    cudaMemcpy(d_points_y, points.y, sizeof(float) * num_points, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_ms_x, sizeof(float));
    cudaMalloc((void**)&d_ms_y, sizeof(float));
    cudaMalloc((void**)&d_num_neighbours, sizeof(int));

    for (int i = 0; i < num_points; i++)
    {
        for (int j = 0; j < MAX_ITERS; j++)
        {
            cudaMemset(d_ms_x, 0, sizeof(float));
            cudaMemset(d_ms_y, 0, sizeof(float));
            cudaMemset(d_num_neighbours, 0, sizeof(int));

            centroid_convergence_kernel_2<<<ceil(num_points / (float)BLOCK_DIM), BLOCK_DIM>>>(&d_centroid_x[i], &d_centroid_y[i], d_points_x, d_points_y, num_points, d_ms_x, d_ms_y, d_num_neighbours);
            cudaDeviceSynchronize();

            float ms_x, ms_y;
            int num_neighbours;
            cudaMemcpy(&ms_x, d_ms_x, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&ms_y, d_ms_y, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_neighbours, d_num_neighbours, sizeof(int), cudaMemcpyDeviceToHost);

            if (num_neighbours > 0)
            {
                ms_x /= num_neighbours;
                ms_y /= num_neighbours;

                // Update centroid position
                centroids.x[i] = ms_x;
                centroids.y[i] = ms_y;
                cudaMemcpy(&d_centroid_x[i], &centroids.x[i], sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(&d_centroid_y[i], &centroids.y[i], sizeof(float), cudaMemcpyHostToDevice);
            }
        }
    }

    cudaMemcpy(centroids.x, d_centroid_x, sizeof(float) * num_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.y, d_centroid_y, sizeof(float) * num_points, cudaMemcpyDeviceToHost);

    cudaFree(d_centroid_x);
    cudaFree(d_centroid_y);
    cudaFree(d_points_x);
    cudaFree(d_points_y);
    cudaFree(d_ms_x);
    cudaFree(d_ms_y);
    cudaFree(d_num_neighbours);
}