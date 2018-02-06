#include <stdio.h>

int gcd(int a, int b) {

	if (a%b == 0) return b;
	else gcd(b, a%b);

} //gcd

void main() {

	int a = 1, b = 1;

	// won't work with large integers (above 2bill)
	// function will run but answers will be meaningless
	printf("Enter the first number: ");
	scanf("%d", &a);
	printf("Enter the second number: ");
	scanf("%d", &b);

	if (gcd(a,b) == 1) {
		printf("%d and %d are coprime.\n",a,b);
	}
	else printf("%d and %d are not coprime.\n",a,b);


} //main
