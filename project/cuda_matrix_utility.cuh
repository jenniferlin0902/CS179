

void call2DConvolveKernal(float* f, float* result, float* input,
                          int input_x, int input_y, int f_x, int f_y);
void callMMSEEstKernal(float* dev_data, int size);
void callWeinerRxyKernal(float* x, float* y, float* Rxy,
                         int x_w, int x_h, int k1, int k2);
void callWeinerUpdateKernal(float* f, float* Rxy, float* Ryy, int f_w, int f_h);
void callFIRNormalizeKernal(float* f, int size);