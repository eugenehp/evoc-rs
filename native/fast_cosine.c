#include <stdint.h>

// Match Numba fast_cosine (float_nndescent.py, fastmath=True).
float fast_cosine_numba(float *x, float *y, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; i++) {
        result += x[i] * y[i];
    }
    if (result > 0.0f) {
        return -result;
    }
    return 1.17549435e-38f;
}
