#include <stdio.h>

int gcd(int a, int b) {
	if (a%b == 0) return b;
	else gcd(b, a%b);
} //gcd

void main() {

	int a = 1, b = 1;
	
	// can't use large integers here - nothing above ~2 bil will fit in an 'int'
	// the program will still run but incorrect numbers will be used (possibly
	// memory address?)
	printf("Enter the first number: ");
	scanf("%d", &a);
	printf("Enter the second number: ");
	scanf("%d", &b);

	int lcm = (a*b) / gcd(a,b);

	printf("The least common multiple of %d and %d is %d.\n", a, b, lcm);


} //main
