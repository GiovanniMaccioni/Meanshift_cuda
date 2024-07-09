#define MAX_ITERS 100000
#define BANDWIDTH 2.0
#define DIM_DATASET 10000
#define BLOCK_DIM 1024
/*------------------------------------------*/

//TODO Ricontrollare se devo mettere i const agli argomenti delle funzioni

/*Data structure to memorize datasets with two features*/
struct dataset2D
{
    float* x;
    float* y;
    int* labels;
};