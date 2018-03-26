#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "functions.h"

int main (int argc, char **argv) {

  MPI_Init(&argc,&argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  //seed value for the randomizer 
  double seed = clock()+rank; //this will make your program run differently everytime
  //double seed = rank; //uncomment this and your program will behave the same everytime it's run

  srand(seed);

  //begin with rank 0 getting user's input
  unsigned int n;

  /* Q3.1 Make rank 0 setup the ELGamal system and
    broadcast the public key information */

  //declare storage
  unsigned int p, g, h, x;
 
  if (rank == 0) {
    printf("Enter a number of bits: "); fflush(stdout);
    char status = scanf("%u",&n);
    
    //make sure the input makes sense
    if ((n<3)||(n>31)) {//Updated bounds. 2 is no good, 31 is actually ok
      printf("Unsupported bit size.\n");
      return 0;   
    }
    printf("\n");

    //setup an ElGamal cryptosystem
    setupElGamal(n,&p,&g,&h,&x);
  }

  unsigned int data[3];
  if (rank == 0) {
    data[0] = p;
    data[1] = g;
    data[2] = h;
  }

  MPI_Bcast(&data, 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    p = data[0];
    g = data[1];
    h = data[2];
  } 

  //Suppose we don't know the secret key. Use all the ranks to try and find it in parallel
  if (rank==0)
    printf("Using %d processes to find the secret key...\n", size);

  double time1 = MPI_Wtime();

  /*Q3.2 We want to loop through values i=0 .. p-2
     determine start and end values so this loop is 
     distributed amounst the MPI ranks  */
  unsigned int N = p-1; //total loop size
  unsigned int start, end;
 
  //divide N / size

  if (N%size == 0) { 
    start = (N/size)*rank; 
    end = start + (N/size);
  }
  else if (rank == (size-1)) {
    start = (1 + (N/size))*rank;
    end = N;
  }
  else {
    start = (1 + (N/size))*rank;
    end = start + (N/size);
  }


  //loop through the values from 'start' to 'end'
  for (unsigned int i=start;i<end;i++) {
    if (modExp(g,i,p)==h)
      printf("Secret key found! x = %u \n", i);
  }

  double time2 = MPI_Wtime();

  printf("It took rank %d %f seconds to iterate %d loops with a throughput of %f \n", rank, time2-time1, end-start, ((end-start)/(time2-time1)));

  MPI_Finalize();

  return 0;
}
