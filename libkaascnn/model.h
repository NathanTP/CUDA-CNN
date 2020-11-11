#ifndef _MODEL_H_
#define _MODEL_H_
#include <stdint.h>
#include "layer.h"

typedef struct {
  layerParams_t *input;
  layerParams_t *c;
  layerParams_t *s;
  layerParams_t *fin;
} modelState_t;

#ifdef __cplusplus
/*=====================================================
 * High level helpers (used by FaaS+GPU configurations)
 *=====================================================
 */
// Create a new model object from layers. You should free() the object returned
// here, but there is no deep-free function, users must manage the layers
// themselves. This is because memory management of layers is complicated and
// up to the user (in kaas, device memory isn't managed by the application).
extern "C" modelState_t *newModel(layerParams_t *input, layerParams_t *c1, layerParams_t *s1, layerParams_t *fin);

// Convenience function to all each layer's forward pass and extract the
// prediction. You can also use the lower-level functions defined below.
extern "C" unsigned int classify(modelState_t *m, float inp[28][28]);

/*====================================================
 * Lowest-layer interfaces, these take only device pointers and only call CUDA
 * kernels. This is as close as we've gotten to the real KaaS interface. In
 * reality, KaaS would be targeted by some sort of DSL that generates these
 * wrappers from user-provided __global__ functions only to ensure safety (KaaS
 * does not run any direct user code on the host). The higher level
 * abstractions above can be thought of as a short-cut until we build the real
 * KaaS infrastructure.
 */
// Input is the image
extern "C" void kaasLayerCForward(float *input, float *preact, float *weight, float *bias, float *output);

// Intermediate layer takes output of layerC
extern "C" void kaasLayerSForward(float *input, float *preact, float *weight, float *bias, float *output);

// Output is the predictions (array of 9 floats with the probability estimates
// for each digit, take the max for the prediction)
extern "C" void kaasLayerFinForward(float *input, float *preact, float *weight, float *bias, float *output);
#else
modelState_t *newModel(layerParams_t *input, layerParams_t *c1, layerParams_t *s1, layerParams_t *fin);
unsigned int classify(modelState_t *m, float inp[28][28]);
void kaasLayerCForward(float *input, float *preact, float *weight, float *bias, float *output);
void kaasLayerSForward(float *input, float *preact, float *weight, float *bias, float *output);
void kaasLayerFinForward(float *input, float *preact, float *weight, float *bias, float *output);
#endif

#endif
