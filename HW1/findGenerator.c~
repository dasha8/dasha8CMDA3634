#include <stdio.h>
#include <math.h>

void main() {

	int p = 1;

	// can't enter really large number here, above 2bil the number can't be stored in int
	// function still runs but with incorrect numbers
	printf("Enter a prime number: ");
	scanf("%d", &p);

	int g = 2, isGen = 0;

	while (isGen == 0 && g < p) {
		isGen = 1;
		for (int r=1;r<p-1;r++) {
			if (((int)pow(g,r))%p == 1) isGen = 0; //if g^r mod p is 1 g isn't generator
		}

		g++;
	
	}

	printf("%d is a generator of Z_%d\n.", g-1, p);

} //main
