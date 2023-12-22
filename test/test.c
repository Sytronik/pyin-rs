#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pyin.h"

int main(void)
{
    unsigned int sr = 24000;
    unsigned int length = sr * 3;
    double *wav = (double *)malloc(sizeof(double) * length);
    unsigned int frame_length = 2048;
    double *timestamp = NULL;
    double *f0 = NULL;
    bool *voiced_flag = NULL;
    double *voiced_prob = NULL;
    unsigned int n_frames = 0;

    for (int i = 0; i < sr; i++)
        wav[i] = sin(2 * M_PI * 240 * i / sr);
    for (int i = sr; i < 2 * sr; i++)
        wav[i] = 0.;
    for (int i = 2 * sr; i < 3 * sr; i++)
        wav[i] = sin(2 * M_PI * 480 * i / sr);

    int result = pyin(&timestamp, &f0, &voiced_flag, &voiced_prob, &n_frames,
                      wav, length, sr, 80., 800., frame_length, 0, 0, 0,
                      NAN, true, 0);

    printf("result: %d\n", result);
    printf("n_frames: %d\n", n_frames);
    if (n_frames > 0)
    {
        printf("timestamp (sec): [");
        for (int i = 0; i < n_frames; i++)
            printf("%.3f, ", timestamp[i]);
        printf("]\n");
        printf("f0 (Hz): [");
        for (int i = 0; i < n_frames; i++)
            printf("%.2f, ", f0[i]);
        printf("]\n");
        printf("voiced_flag: [");
        for (int i = 0; i < n_frames; i++)
        {
            if (voiced_flag[i])
                printf("true, ");
            else
                printf("false, ");
        }
        printf("]\n");
        printf("voiced_prob: [");
        for (int i = 0; i < n_frames; i++)
            printf("%.2f, ", voiced_prob[i]);
        printf("]\n");
    }

    free(wav);
    free(timestamp);
    free(f0);
    free(voiced_flag);
    free(voiced_prob);
    return 0;
}
