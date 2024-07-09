#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "math.h"
#include <omp.h>
#include <chrono>
#include "structures.h"


/*---------------------DATA AND MEMORY MANAGING FUNCTIONS-------------------------------*/
//Function to copy an array.
//TOCHECK stiamo parlando di talvolta dimensione campionarie molto alte. La copia non so quanto possa essere performante e/o efficace
inline float* copy_array(float* target, int dim)
{
    float* copy = (float*)malloc(sizeof(float)*dim);
    memcpy(copy, target, sizeof(float)*dim);
    return copy;  
}

//TODO aggiungere errore di lettura/scrittura per terminare il programma
//Function to upload the dataset in memory
void upload_dataset2D(dataset2D &d, std::string fname)
{
	std::string line, word;
 
	std::fstream file (fname, std::ios::in);
	if(file.is_open())
	{
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

            //if(file.eof() == 1)
            //    break;
		}
	}
	else
		std::cout<<"Could not open the file\n";
}

//Function to write the output
void write_csv(dataset2D c, dataset2D d, int o_d_dim, std::string fname)
{
    std::fstream file (fname, std::ios::out);
    
    for(int i = 0; i < o_d_dim; i++)
        file << d.x[i] << ',' << d.y[i] << ',' << c.labels[i] << '\n';

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