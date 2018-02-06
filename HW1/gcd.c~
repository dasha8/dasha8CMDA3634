#include <stdio.h>

int gcd(int a, int b) {

	if (a%b == 0) return b;
	else gcd(b, a%b);

} //gcd

void main() {

	int a = 1, b = 1; //because we are declaring a and b as ints, they are limited in size
			  //numbers higher than ~2billion aren't stored and computed correctly
			  //in this instance the function will run, but answers are meaningless

	//input numbers
	printf("Enter the first number: ");
	scanf("%d", &a);
	printf("Enter the second number: ");
	scanf("%d", &b);

	//calc and display greatest common divisor
	int c = gcd(a,b);
	printf("The greatest common divisor of %d and %d is %d.\n",a,b,c);


} //main
