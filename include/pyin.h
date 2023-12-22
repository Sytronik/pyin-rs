#ifndef PYIN
#define PYIN

#include <stdbool.h>

// The caller must call free on f0, voiced_flag, voiced_prob to
// prevent a memory leak.
int pyin(
    // outputs
    double **timestamp,
    double **f0,
    bool **voiced_flag,
    double **voiced_prob,
    unsigned int *n_frames,

    // inputs
    const double *input,
    unsigned int length,
    unsigned int sr,
    double fmin,
    double fmax,
    unsigned int frame_length,
    unsigned int win_length, // If 0, use default value (frame_length / 2)
    unsigned int hop_length, // If 0, use default value (frame_length / 4)
    double resolution,       // 0 < resolution < 1. If <= 0, use default value (0.1)
    double fill_unvoiced,
    bool center,
    unsigned int pad_mode); // 0: zero padding, 1: reflect padding

#endif
