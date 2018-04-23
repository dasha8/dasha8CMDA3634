#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "cuda.h"
#include "functions.c"

//compute a*b mod p safely
__device__ unsigned int kernelModprod(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int za = a;
  unsigned int ab = 0;

  while (b > 0) {
    if (b%2 == 1) ab = (ab +  za) % p;
    za = (2 * za) % p;
    b /= 2;
  }
  return ab;
}

//compute a^b mod p safely
__device__ unsigned int kernelModExp(unsigned int a, unsigned int b, unsigned int p) {
  unsigned int z = a;
  unsigned int aExpb = 1;

  while (b > 0) {
    if (b%2 == 1) aExpb = kernelModprod(aExpb, z, p);
    z = kernelModprod(z, z, p);
    b /= 2;
  }
  return aExpb;
}

__global__ void findKey(unsigned int p, unsigned int g, unsigned int h, unsigned int* d_x) {

  int thread = threadIdx.x;
  int block = blockIdx.x;
  int Nblock = blockDim.x;

  int id = Nblock*block + thread;

  if (id < (p-1)) {  
    if (kernelModExp(g, id+1, p) == h) {
      d_x* = id+1;
    }
  }

}


}

int main (int argc, char **argv) {

  /* Part 2. Start this program by first copying the contents of the main function from 
     your completed decrypt.c main function. */

  /* Q4 Make the search for the secret key parallel on the GPU using CUDA. */

  //declare storage for an ElGamal cryptosytem
  unsigned int n, p, g, h, x;
  unsigned int Nints;

  //get the secret key from the user
  printf("Enter the secret key (0 if unknown): "); fflush(stdout);
  char stat = scanf("%u",&x);

  printf("Reading file.\n");

  /* Q3 Complete this function. Read in the public key data from public_key.txt
    and the cyphertexts from messages.txt. */

  FILE *f = fopen("public_key.txt", "r");
  fscanf(f, "%u %u %u %u", &n, &p, &g, &h);
  fclose(f);

  f = fopen("message.txt", "r");
  fscanf(f, "%u", &Nints);

  unsigned int *Zmessage =
      (unsigned int *) malloc(Nints*sizeof(unsigned int));

  unsigned int *a =
      (unsigned int *) malloc(Nints*sizeof(unsigned int));

  for (unsigned int i=0; i<Nints;i++) {
    fscanf(f, "%u %u", &(Zmessage[i]), &(a[i]));
  }

  fclose(f);

  // find the secret key
  if (x==0 || modExp(g,x,p)!=h) {
    printf("Finding the secret key...\n");

    double startTime = clock();

    unsigned int h_x;
    unsigned int *d_x;
    cudaMalloc(&d_x, sizeof(unsigned int));
    
    int Nthreads = 32;
    int Nblocks = (p - 1)/Nthreads;

    findKey <<<Nblocks, Nthreads>>> (p, g, h);

    cudaDeviceSynchronize();

    cudaMemcpy(h_x, d_x, 4*sizeof(int), cudaMemcpyDeviceToHost);

    x = h_x;

    cudaFree(d_x);
  }

  double endTime = clock();

  double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
  double work = (double) p;
  double throughput = work/totalTime;

  printf("The key found is %u\n", x);
  printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);
  return 0;
}
