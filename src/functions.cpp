#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "math.h"
#include <omp.h>
#include <chrono>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions.hpp>



/*---------------------DATA AND MEMORY MANAGING FUNCTIONS-------------------------------*/
//Function to copy an array.
inline float* copy_array(float* target, int dim)
{
    float* copy = (float*)malloc(sizeof(float)*dim);
    memcpy(copy, target, sizeof(float)*dim);
    return copy;  
}

//Function to upload the dataset in memory
void upload_dataset2D(dataset2D &d, std::string fname)
{
	std::string line, word;
 
	std::fstream file (fname, std::ios::in);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << fname << std::endl;
        return;
    }
    int counter = 0;
    while(std::getline(file, line, '\n'))
    {
        std::stringstream str(line);
        int component = 0;

        while(std::getline(str, word, ','))
        {
            if(component == 0)
                d.x[counter] = std::stof(word);
            else if(component == 1)
                d.y[counter] = std::stof(word);
            else
                d.labels[counter] = std::stoi(word);
        
            component++;
            //std::cout << stof(word) << "\t";
        }
        //std::cout << std::endl;
        counter++;
    }
}

//Function to write the dataset to csv
void write_csv(dataset2D d, int o_d_dim, std::string fname)
{
    std::fstream file (fname, std::ios::out);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << fname << std::endl;
        return;
    }
    
    for(int i = 0; i < o_d_dim; i++)
        file << d.x[i] << ',' << d.y[i] << ',' << d.labels[i] << '\n';

    file.close();
}

void print_dataset2D(dataset2D d, int d_dim)
{
    for(int i = 0; i < d_dim; i++)
    {
        std::cout << d.x[i]<< "\t" << d.y[i] << "\t" << d.labels[i] << std::endl;
    }
    std::cout << std::endl;
}


void generate_2d_blobs(
    dataset2D &data,
    float* means_x,
    float* means_y,
    float* std_x,
    float* std_y,
    int samples_per_cluster,
    int num_clusters   
) {
    data.x = (float*)malloc(sizeof(float)*samples_per_cluster*num_clusters);
    data.y = (float*)malloc(sizeof(float)*samples_per_cluster*num_clusters);
    data.labels = (int*)malloc(sizeof(int)*samples_per_cluster*num_clusters);

    boost::random::mt19937 rng;
    for (int cluster = 0; cluster < num_clusters; ++cluster) 
    {
        boost::normal_distribution<float> dist_x(means_x[cluster], std_x[cluster]);
        boost::normal_distribution<float> dist_y(means_y[cluster], std_y[cluster]);

        for (int i = 0; i < samples_per_cluster; ++i)
        {
            data.x[samples_per_cluster*cluster + i] = dist_x(rng);
            data.y[samples_per_cluster*cluster + i] = dist_y(rng);
            data.labels[samples_per_cluster*cluster + i] = cluster;
        }
    }
}

void merge_cluster2D(dataset2D &c, int o_d_dim, float b)
{
    int l_counter = 0;
    for(int c_index = 0; c_index < o_d_dim; c_index++)
    {
        //DEBUG
        //std::cout << "inside for(int i = 0; i < o_d_dim; i++)" << std::endl;
        //
        //if the centroid hasn't been altready labeled, assign a label and advance the label counter to start a new cluster
        if(c.labels[c_index] == -1)
        {
            //#pragma omp parallel for num_threads(NUM_THREADS)
            for(int j = 0; j < o_d_dim; j++)
            {
                //if the centroid hasn't been labed already
                if(c.labels[j] == -1)
                {
                    if(is_neighbour2D(c.x[c_index], c.y[c_index], c.x[j], c.y[j], b))
                        c.labels[j] = l_counter;
                }
            }
            l_counter++;
        }
    }
}


