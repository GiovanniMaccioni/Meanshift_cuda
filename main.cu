#include "include/structures.h"
#include "src/functions_cu.cu"
#include "src/functions.cpp"


int main()
{
    dataset2D data;
    dataset2D centroids;
    
    std::string file_csv_in;
    std::string file_csv_out;
    std::string dataset_path;

    dataset_path = "dataset/blobs_5clusters_1000000samples";
    file_csv_out = dataset_path + "_output.csv";
    file_csv_in = dataset_path +".csv";

    data.x = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.y = (float*)malloc(sizeof(float)*DIM_DATASET);
    data.labels = (int*)malloc(sizeof(int)*DIM_DATASET);

    upload_dataset2D(data, file_csv_in);
    //print_dataset2D(data, DIM_DATASET);

    //Here we initialize the centroid data structure, that will be modified throughout the algorithm
    //For starters the centroids are all the points in the dataset
    centroids.x = copy_array(data.x, DIM_DATASET);
    centroids.y = copy_array(data.y, DIM_DATASET);
    centroids.labels = new int[DIM_DATASET];
    /*Centroid Labels inizialization*/
    for(int i = 0; i < DIM_DATASET; i++)
        centroids.labels[i] = -1;


    auto start_conv = std::chrono::high_resolution_clock::now();

    //Uncomment for M1, Comment for M2
    //meanshift_convergence(centroids, data, DIM_DATASET);

    //Uncomment for M2, Comment for M1
    meanshift_convergence_2(centroids, data, DIM_DATASET);
    
    auto stop_conv = std::chrono::high_resolution_clock::now();
    auto duration_conv = std::chrono::duration_cast<std::chrono::milliseconds>(stop_conv - start_conv);

    printf("Work took %ld milliseconds\n", duration_conv.count());

    auto start = std::chrono::high_resolution_clock::now();
    
    //meanshift_merge(centroids, DIM_DATASET);
    merge_cluster2D(centroids, DIM_DATASET, BANDWIDTH);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("Work took %ld milliseconds\n", duration.count());

    for(int i = 0; i < DIM_DATASET; i++)
        data.labels[i] = centroids.labels[i];

    write_csv(data, DIM_DATASET, file_csv_out);
}