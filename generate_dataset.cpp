#include "include/structures.h"
#include "include/defines.h"
#include "src/functions.cpp"

int main() {
    dataset2D data;
    std::string dataset_path;
    std::string file_csv;
    std::string datafile;
    std::string scatter_plot;

    dataset_path = "./dataset/blobs_5clusters_1000000samples";
    file_csv = dataset_path+".csv";
    scatter_plot = dataset_path+"_scatter.png";


    int num_clusters = 5; 
    float means_x[num_clusters] = {0., 3., 3., -3., -3.};
    float means_y[num_clusters] = {0., 3., -3., -3., 3.};
    float std_x[num_clusters] = {1., 1., 1., 1., 1.};
    float std_y[num_clusters] = {1., 1., 1., 1., 1.};
    int samples_per_cluster = 200000;

    generate_2d_blobs(data, means_x, means_y, std_x, std_y, samples_per_cluster, num_clusters);
    write_csv(data, samples_per_cluster*num_clusters, file_csv);

    //scatterplot_gnu(file_csv, scatter_plot);

    return 0;
}