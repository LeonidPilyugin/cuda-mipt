#include <stdio.h>

int main() {
    for (int sigma = 0; sigma < 26; sigma++) {
        printf("%d: ", sigma);
        for (int f = 0; f < 2; f++) {
            int result = sigma / (2 * f + 2) - (3 - 2 * f);
            printf("%d ", result==0);
        }    
        printf("\n");
    }
    
    return 0;
}
