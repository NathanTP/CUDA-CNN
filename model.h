#include "layer.h"
#include <string>

#ifndef MODEL_H
#define MODEL_H

class Model {
  public:
  Layer *l_input;
  Layer *l_c1;
  Layer *l_s1;
  Layer *l_f;

  Model();
  Model(std::string modelDir, bool enableTrain);
  ~Model();

  bool save(std::string modelDir);
};

#endif
