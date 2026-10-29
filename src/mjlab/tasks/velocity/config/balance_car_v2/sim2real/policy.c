#include <math.h>
#include "model_weights.h"

static inline float elu(float x) { 
    if (x >= 0.0f) { 
        return x; 
    } 
    return 1.0f * (expf(x) - 1.0f); 
}

static void normalize_obs(const float *raw_obs, float *norm_obs) {
    for (int i = 0; i < 12; i++) {
        norm_obs[i] = (raw_obs[i] - obs_normalizer_mean[i]) / obs_normalizer_std[i];
    }
}

static void matrix_vec_mult(const float *x, const float *w, const float *b, float *out, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = b[i];
        for (int j = 0; j < in_dim; j++) {
            sum += x[j] * w[i * in_dim + j];
        }
        out[i] = sum;
    }
}

void model_forward(const float *raw_obs, float *action) {
    float x0[12];
    float x1[24];
    float x2[12];

    normalize_obs(raw_obs, x0);
    matrix_vec_mult(x0, mlp_0_weight, mlp_0_bias, x1, 12, 24);
    for (int i = 0; i < 24; i++) {
        x1[i] = elu(x1[i]);
    }
    matrix_vec_mult(x1, mlp_2_weight, mlp_2_bias, x2, 24, 12);
    for (int i = 0; i < 12; i++) {
        x2[i] = elu(x2[i]);
    }
    matrix_vec_mult(x2, mlp_4_weight, mlp_4_bias, action, 12, 2);
}

void main() {
    float raw_obs[12] = {-1.0321e-02, -1.7328e-01,  3.8765e-01,  2.6002e-02,  4.7311e-05, -9.9966e-01,  8.3297e+00,  6.8149e+00,  5.3278e-01, -5.7659e-01, -4.4091e-01,  3.7685e-01};
    float action[2] = {0};
    model_forward(raw_obs, action);

    return;
}

