#include "include_cpp/functions.h"
#include "include_cuda/functions.cu"

int main()
{
    dataset2D data;
    dataset2D centroids;
    
    std::string file_csv_in;
    std::string file_csv_out;
    std::string dataset_path;

    dataset_path = "./dataset/data10000";
    file_csv_out = file_csv_in + "_output.csv";
    file_csv_in = file_csv_in+".csv";

    data.x = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.y = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.labels = (int*)malloc(sizeof(int)*DIM_DATASET);

    upload_dataset2D(data, dataset_path+file_csv_in);
    //print_dataset2D(data, DIM_DATASET);

    //Here we initialize the centroid data structure, that will be modified throughout the algorithm
    //For starters the centroids are all the points in the dataset
    centroids.x = copy_array(data.x, DIM_DATASET);
    centroids.y = copy_array(data.y, DIM_DATASET);
    centroids.labels = new int[DIM_DATASET];
    /*Centroid Labels inizialization*/
    for(int i = 0; i < DIM_DATASET; i++)
        centroids.labels[i] = -1;

    //std::cout << "------CENTROIDS--------" << std::endl;
    //print_dataset2D(centroids, DIM_DATASET);

    auto start_conv = std::chrono::high_resolution_clock::now();
        //points are to be saved in constant memory

    //Move points in global memory

    //for(int i = 0; i < DIM_DATASET; ++i)
    //    meanshift_convergence(&centroids.x[i], &centroids.y[i], data, DIM_DATASET);
    meanshift_convergence(centroids, data, DIM_DATASET);
    
    auto stop_conv = std::chrono::high_resolution_clock::now();
    auto duration_conv = std::chrono::duration_cast<std::chrono::milliseconds>(stop_conv - start_conv);

    printf("Work took %d milliseconds\n", duration_conv.count());

    //std::cout << "------UPDATED CENTROIDS--------" << std::endl;
    //print_dataset2D(centroids, DIM_DATASET);

    auto start = std::chrono::high_resolution_clock::now();
    
    //We supposed that, given epsilon small enough, if c1 is close to c2 and c2 to c3, then c1 is close to c3
    meanshift_merge(centroids, DIM_DATASET);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("Work took %d milliseconds\n", duration.count());
    //print_dataset2D(centroids, DIM_DATASET);

    write_csv(centroids, data, DIM_DATASET, dataset_path+file_csv_out);
}