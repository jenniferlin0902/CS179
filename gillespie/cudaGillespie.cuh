

void callGillepsieKernal(
        int nTrials
        const float k_on,
        const float k_off,
        const float g,
        const float b,
        int* state,
        float* X,
        float* rand1,
        unsigned int threadsPerBlock,
        unsigned int blocks);
