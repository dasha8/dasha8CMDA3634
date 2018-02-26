#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "functions.h"

int main (int argc, char **argv) {

	//seed value for the randomizer 
  double seed;
  
  seed = clock(); //this will make your program run differently everytime
  //seed = 0; //uncomment this and you program will behave the same everytime it's run
  
  srand48(seed);


  //begin by getting user's input
	unsigned int n;

  printf("Enter a number of bits: ");
  scanf("%u",&n);

  //make sure the input makes sense
  if ((n<2)||(n>30)) {
  	printf("Unsupported bit size.\n");
		return 0;  	
  }

  int p = randXbitInt(n);

  /* Q2.2: Use isProbablyPrime and randomXbitInt to find a random n-bit prime number */

  while (!isProbablyPrime(p)) {
    p = randXbitInt(n); 
  }

  printf("p = %u is probably prime.\n", p);

  /* Q3.2: Use isProbablyPrime and randomXbitInt to find a new random n-bit prime number 
     which satisfies p=2*q+1 where q is also prime */
  int q = (p-1)/2;
  while (isProbablyPrime(p) == 0 || isProbablyPrime(q) == 0) {
    p = randXbitInt(n);
    q = (p-1)/2;
  }
	printf("p = %u is probably prime and equals 2*q + 1. q= %u and is also probably prime.\n", p, q);  

	/* Q3.3: use the fact that p=2*q+1 to quickly find a generator */
	unsigned int g = findGenerator(p);

	printf("g = %u is a generator of Z_%u \n", g, p);  

  /* Bonus */
  unsigned int x = (int)(rand()%p);
  unsigned int h = (unsigned int)(pow(g,x));

  printf("%u is our initial x and %u is h.\n", x, h);
    
  for (int pos=1;pos<=p;pos++) {
    if ((int)pow(g,pos) == h) {
      x = pos;
      break;
    }
  }

  printf("%u is the value of x.\n", x);

  return 0;
}
