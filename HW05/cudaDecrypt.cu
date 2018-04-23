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

__global__ void findKey(*d_values) {

  int thread = threadIdx.x;
  int block = blockIdx.x;
  int Nblock = blockDim.x;

  int id = thread + block*Nblock;

  unsigned int p = d_value[0];
  unsigned int g = d_value[1];
  unsigned int h = d_value[2];

  for (unsigned int i=id; i<id+Nblock; i++) { //HOW MANY ITER
    if (kernelModExp(g, i+1, p) == h) {
      x = i+1;
      d_values[3] = x;
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

    int *h_values = malloc(4*sizeof(int));
    h_value[0] = p;
    h_value[1] = g;
    h_value[2] = h;
    h_value[3] = 0; //space for x once found
    int *d_values;
    cudaMalloc(&d_values, 4*sizeof(int));
    
    cudaMemcpy(d_values,h_values,4*sizeof(int),cudaMemcpyHostToDevice);

    int Nthreads = 32;
    int Nblocks = (Nthreads + p - 1)/Nthreads;

    kernelSecretKey<<<Nblocks, Nthreads>>>(d_values);

    cudaDeviceSynchronize();

    cudaMemcpy(h_values, d_values, 4*sizeof(int), cudaMemcpyDeviceToHost);

    x = d_values[3];

    cudaFree(d_values);
    free(h_values);

  }

  double endTime = clock();

  double totalTime = (endTime-startTime)/CLOCKS_PER_SEC;
  double work = (double) p;
  double throughput = work/totalTime;

  printf("Searching all keys took %g seconds, throughput was %g values tested per second.\n", totalTime, throughput);
  return 0;
}
