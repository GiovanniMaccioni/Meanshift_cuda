#include <cuda.h>
#include<cuda_runtime.h>
//#include "../include_cpp/structures.h"


__device__
inline float euclidian_norm2D(float x, float y)
{
    return sqrt(pow(x,2) + pow(y,2));
}

__device__
inline bool is_neighbour2D(float c_x, float c_y, float p_x, float p_y, float bandwidth)
{
    //DEBUG
    //std::cout << "distance: " << euclidian_norm2D(c_x - p_x, c_y - p_y)<<std::endl;
    //
    return euclidian_norm2D(c_x - p_x, c_y - p_y) < bandwidth? true:false;
}



__global__
void centroid_convergence_kernel(float* d_centroid_x, float* d_centroid_y, float* d_points_x, float* d_points_y, int num_points)
{
    int thread_id = blockDim.x*blockIdx.x + threadIdx.x;
    bool check = false;//automatic variable
    //float bandwidth = BANDWIDTH;
    float meanshift_x = 0;
    float meanshift_y = 0;
    int num_neighbours = 0;
    if(thread_id < num_points)
    {    
        for(int i = 0; i < num_points; i++)
        {
            check = is_neighbour2D(d_centroid_x[thread_id], d_centroid_y[thread_id], d_points_x[i], d_points_y[i], BANDWIDTH);
            if(check)
            {
                meanshift_x += d_points_x[i];
                meanshift_y += d_points_y[i];
                num_neighbours += 1;
            }
        }

        d_centroid_x[thread_id] = meanshift_x / num_neighbours;
        d_centroid_y[thread_id] = meanshift_y / num_neighbours;
    }
}




void meanshift_convergence(dataset2D centroids, dataset2D points, int num_points)
{
    float *d_centroid_x;
    float *d_centroid_y;

    float *d_points_x;
    float *d_points_y;

    float* d_meanshift_x;
    float* d_meanshift_y;

    int* d_num_neighbours;

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
void merge_clusters_kernel(float* d_centroid_xi, float* d_centroid_yi, float* d_centroid_x, float* d_centroid_y, int* d_points_labels, int num_points, int* label)
{
    int thread_id = blockDim.x*blockIdx.x + threadIdx.x;
    bool check = false;
    float bandwidth = BANDWIDTH;
    if (thread_id < num_points)
    {
        if(d_points_labels[thread_id] == -1)
            check = is_neighbour2D(*d_centroid_xi, *d_centroid_xi, d_centroid_x[thread_id], d_centroid_y[thread_id], bandwidth);
            if (check)
                d_points_labels[thread_id] = *label;
        
        /*__syncthreads();
        //NO
        if(thread_id == 0)
            *label += 1;*/
    }
}

void meanshift_merge(dataset2D centroids, int num_points)
{
    float* d_centroid_x;
    float* d_centroid_y;
    int* d_centroids_labels;

    int* d_label;

    cudaMalloc((void**) &d_centroid_x, sizeof(float)*num_points);
    cudaMemcpy(d_centroid_x, centroids.x, sizeof(float)*num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_centroid_y, sizeof(float)*num_points);
    cudaMemcpy(d_centroid_y, centroids.y, sizeof(float)*num_points, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_centroids_labels, sizeof(int)*num_points);
    cudaMemcpy(d_centroids_labels, centroids.labels, sizeof(int)*num_points, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_label, sizeof(int));
    cudaMemset(d_label, 0, sizeof(int));

    //KERNEL CALL
    //We will have a 1d block
    for(int i = 0; i < num_points; i++)
    {
        merge_clusters_kernel<<<ceil(num_points/1024.0), 1024>>>(&d_centroid_x[i], &d_centroid_y[i], d_centroid_x, d_centroid_y, d_centroids_labels, num_points, d_label);
        //*d_label = *d_label + 1;
    }

    cudaMemcpy(centroids.labels, d_centroids_labels, sizeof(float)*num_points, cudaMemcpyDeviceToHost);

    cudaFree(d_centroid_x);
    cudaFree(d_centroid_y);
    cudaFree(d_centroids_labels);
    cudaFree(d_label);

    return;
}