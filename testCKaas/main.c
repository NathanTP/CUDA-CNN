#include <stdio.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#include "libkaascnn.h"

typedef struct {
  mnist_data *train_set;
  mnist_data *test_set;
  unsigned int train_cnt;
  unsigned int test_cnt;
} mnistData_t;

static inline mnistData_t *loaddata(void)
{
  int err;

  mnistData_t *dat = malloc(sizeof(mnistData_t));

	err = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte",
		&dat->train_set, &dat->train_cnt);
  if(err != 0) {
    fprintf(stderr, "Failed to load training data\n");
    return NULL;
  }
	err = mnist_load("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte",
		&dat->test_set, &dat->test_cnt);
  if(err != 0) {
    fprintf(stderr, "Failed to load t10k-images data\n");
    return NULL;
  }
  return dat;
}

int main(void)
{
  bool ok;
  ok = initLibkaascnn();
  if(!ok) {
    return 1;
  }

  // This is just hard coded for now, the model was trained using libfaascnn
  layerParams_t *hostLayers[4];
  hostLayers[0] = layerParamsFromFile("./testModel/l_input");
  hostLayers[1] = layerParamsFromFile("./testModel/l_c1");
  hostLayers[2] = layerParamsFromFile("./testModel/l_s1");
  hostLayers[3] = layerParamsFromFile("./testModel/l_f");

  // A KaaS system would manage inputs and outputs outside the user code, in
  // this case we're pretending to be the provider and loading the data before
  // running the model.
  layerParams_t *devLayers[4];
  for(int i = 0; i < 4; i++) {
    if(hostLayers[i] == NULL) {
      fprintf(stderr, "Layer %d did not load from file\n", i);
      return 1;
    }
    
    devLayers[i] = layerParamsToDevice(hostLayers[i]);
    if(devLayers[i] == NULL) {
      fprintf(stderr, "Layer %d did not load onto the device\n", i);
      return 1;
    }    
  }

  modelState_t *m = newModel(devLayers[0], devLayers[1], devLayers[2], devLayers[3]);
  if(!m) {
    return 1;
  }

  mnistData_t *dat = loaddata();
  if(!dat) {
    return 1;
  }

  // No batching right now
  int nerr = 0;
  for(unsigned int i = 0; i < dat->test_cnt; i++) {
  /* for(unsigned int i = 0; i < 2; i++) { */
    unsigned int pred = classify(m, dat->test_set[i].data);
    if(pred != dat->test_set[i].label) {
      /* printf("wrong pred (input %d): %d != %d\n", i, pred, dat->test_set[i].label); */
      nerr++;
    }
  }

  printf("Done, error: %lf\n", (double)nerr / dat->test_cnt);
  return 0;
}
