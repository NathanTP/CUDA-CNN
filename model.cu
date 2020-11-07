#include "model.h"
#include <sys/stat.h>
#include <string.h>

Model::Model(void)
{
  l_input = new Layer(0, 0, 28*28);
  l_c1 = new Layer(5*5, 6, 24*24*6);
  l_s1 = new Layer(4*4, 1, 6*6*6);
  l_f = new Layer(6*6*6, 10, 10);
}

Model::Model(std::string modelDir, bool enableTrain)
{
  l_input = new Layer(modelDir + "/l_input", enableTrain);
  l_c1 = new Layer(modelDir + "/l_c1", enableTrain);
  l_s1 = new Layer(modelDir + "/l_s1", enableTrain);
  l_f = new Layer(modelDir + "/l_f", enableTrain);
}

Model::~Model()
{
  delete l_input;
  delete l_c1;
  delete l_s1;
  delete l_f;
}

bool Model::save(std::string modelDir)
{
  errno = 0;
  if(mkdir(modelDir.c_str(), 0700 ) != 0) {
    if(errno != EEXIST) {
      printf("Failed to save model: %s", strerror(errno));
      return false;
    }
  }


  if(!l_input->save(modelDir + "/l_input")) {
    printf("Failed to save model\n");
    return false;
  }

  if(!l_c1->save(modelDir + "/l_c1")) {
    printf("Failed to save model\n");
    return false;
  }

  if(!l_s1->save(modelDir + "/l_s1")) {
    printf("Failed to save model\n");
    return false;
  }

  if(!l_f->save(modelDir + "/l_f")) {
    printf("Failed to save model\n");
    return false;
  }

  return true;
}
