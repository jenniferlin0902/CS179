#include <iostream>

using namespace std;
void test5() {
    int *a = (int *) malloc(sizeof (int));
    printf("please enter a number\n");
    scanf("%d", a);
    if (!(*a))
        printf("Value is 0\n");
}

int main() {
    test5();
    return 0;
}