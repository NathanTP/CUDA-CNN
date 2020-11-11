#include "model.h"
#include "kernels.h"
#include "cuda.h"

/*=====================================================
 * High level helpers (used by FaaS+GPU configurations)
 *=====================================================
 */
extern "C" modelState_t *newModel(layerParams_t *input, layerParams_t *c1, layerParams_t *s1, layerParams_t *fin)
{
  modelState_t *m = (modelState_t*)malloc(sizeof(modelState_t));
  m->input = input;
  m->c = c1;
  m->s = s1;
  m->fin = fin;
  return m;
}

extern "C" unsigned int classify(modelState_t *m, double inp[28][28])
{
  clearLayer(m->input);
  clearLayer(m->c);
  clearLayer(m->s);
  clearLayer(m->fin);

  float finp[28][28];
  for(int i = 0; i < 28; i++) {
    for(int j = 0; j < 28; j++) {
      finp[i][j] = (float)inp[i][j];
    }
  }
  cudaMemcpy(m->input->output, finp, 28*28*sizeof(float), cudaMemcpyHostToDevice);

  kaasLayerCForward(m->input->output, m->c->preact, m->c->weight, m->c->bias, m->c->output);
  kaasLayerSForward(m->c->output, m->s->preact, m->s->weight, m->s->bias, m->s->output);
  kaasLayerFinForward(m->s->output, m->fin->preact, m->fin->weight, m->fin->bias, m->fin->output);

  float res[10] = {0};
  cudaMemcpy(res, m->fin->output, 10*sizeof(float), cudaMemcpyDeviceToHost);
  
  int max = 0;
	for (int i = 0; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

  return max;
}

/*=====================================================
 * Lowest level interface (used by KaaS)
 *=====================================================
*/
// Input is the image
extern "C" void kaasLayerCForward(float *input, float *preact, float *weight, float *bias, float *output)
{
	fp_preact_c1<<<64, 64>>>((float (*)[28])input, (float (*)[24][24])preact, (float (*)[5][5])weight);
	fp_bias_c1<<<64, 64>>>((float (*)[24][24])preact, bias);
	apply_step_function<<<64, 64>>>(preact, output,  24*24*6);
}

// Intermediate layer takes output of layerC
extern "C" void kaasLayerSForward(float *input, float *preact, float *weight, float *bias, float *output)
{
	fp_preact_s1<<<64, 64>>>((float (*)[24][24])input, (float (*)[6][6])preact, (float (*)[4][4])weight);
	fp_bias_s1<<<64, 64>>>((float (*)[6][6])preact, bias);
	apply_step_function<<<64, 64>>>(preact, output, 6*6*6);
}

// Output is the predictions (array of 9 floats with the probability estimates
// for each digit, take the max for the prediction)
extern "C" void kaasLayerFinForward(float *input, float *preact, float *weight, float *bias, float *output)
{
	fp_preact_f<<<64, 64>>>((float (*)[6][6])input, preact, (float (*)[6][6][6])weight);
	fp_bias_f<<<64, 64>>>(preact, bias);
	apply_step_function<<<64, 64>>>(preact, output, 10);
}
