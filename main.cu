#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"
#include "model.h"

#include <iostream>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
/* static Layer l_input = Layer(0, 0, 28*28); */
/* static Layer l_c1 = Layer(5*5, 6, 24*24*6); */
/* static Layer l_s1 = Layer(4*4, 1, 6*6*6); */
/* static Layer l_f = Layer(6*6*6, 10, 10); */

static void learn(Model *m);
static unsigned int classify(double data[28][28], Model *m);
static void test(Model *m);
static double forward_pass(double data[28][28], Model *m);
static double back_pass(Model *m);

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
  Model *m;
  if(argc > 1) {
    m = new  Model(std::string(argv[1]), true); 
  } else {
    m = new Model();
  }
	srand(time(NULL));

	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}

  loaddata();
  if(argc > 1) {
    test(m);
  } else {
    learn(m);
    m->save("testModel");
    test(m);
  }

  delete m;
	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28], Model *m)
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	m->l_input->clear();
	m->l_c1->clear();
	m->l_s1->clear();
	m->l_f->clear();

	clock_t start, end;
	start = clock();

	m->l_input->setOutput((float *)input);
	
	fp_preact_c1<<<64, 64>>>((float (*)[28])m->l_input->output, (float (*)[24][24])m->l_c1->preact, (float (*)[5][5])m->l_c1->weight);
	fp_bias_c1<<<64, 64>>>((float (*)[24][24])m->l_c1->preact, m->l_c1->bias);
	apply_step_function<<<64, 64>>>(m->l_c1->preact, m->l_c1->output, m->l_c1->O);

	fp_preact_s1<<<64, 64>>>((float (*)[24][24])m->l_c1->output, (float (*)[6][6])m->l_s1->preact, (float (*)[4][4])m->l_s1->weight);
	fp_bias_s1<<<64, 64>>>((float (*)[6][6])m->l_s1->preact, m->l_s1->bias);
	apply_step_function<<<64, 64>>>(m->l_s1->preact, m->l_s1->output, m->l_s1->O);

	fp_preact_f<<<64, 64>>>((float (*)[6][6])m->l_s1->output, m->l_f->preact, (float (*)[6][6][6])m->l_f->weight);
	fp_bias_f<<<64, 64>>>(m->l_f->preact, m->l_f->bias);
	apply_step_function<<<64, 64>>>(m->l_f->preact, m->l_f->output, m->l_f->O);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

/* XXX Some thoughts about how this might look. I don't know if the generic_forward thing will work, it might need some extra args to nvcc  */
#if 0
/* static double pyplover_forward_pass(double data[28][28]) */
/* { */
/*   /* PACK INPUT INTO s */ */
/*   /* res = pyplover_call("generic_forward"); */ */
/*   generic_forward<<<64, 64>>>(s) */
/*   /* get outputs and put into res*/ */
/*   return res */
/* } */
/*  */
/* __global__ genric_forward(state_t *s) */
/* { */
/*   /*  */
/*      UNPACK ARGUMENTS FROM S */
/*   */ */
/* 	fp_preact_c1<<<64, 64>>>((float (*)[28])l_input->output, (float (*)[24][24])l_c1->preact, (float (*)[5][5])l_c1->weight); */
/* 	fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1->preact, l_c1->bias); */
/* 	apply_step_function<<<64, 64>>>(l_c1->preact, l_c1->output, l_c1->O); */
/*  */
/* 	fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1->output, (float (*)[6][6])l_s1->preact, (float (*)[4][4])l_s1->weight); */
/* 	fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1->preact, l_s1->bias); */
/* 	apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O); */
/*  */
/* 	fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1->output, l_f->preact, (float (*)[6][6][6])l_f->weight); */
/* 	fp_bias_f<<<64, 64>>>(l_f->preact, l_f->bias); */
/* 	apply_step_function<<<64, 64>>>(l_f->preact, l_f->output, l_f->O); */
/* 	 */
/*   /* */
/*      PACK OUTPUT INTO s.out */
/*  */ */
/* } */
#endif

// Back propagation to update weights
static double back_pass(Model *m)
{
	clock_t start, end;

	start = clock();

	bp_weight_f<<<64, 64>>>((float (*)[6][6][6])m->l_f->d_weight, m->l_f->d_preact, (float (*)[6][6])m->l_s1->output);
	bp_bias_f<<<64, 64>>>(m->l_f->bias, m->l_f->d_preact);

	bp_output_s1<<<64, 64>>>((float (*)[6][6])m->l_s1->d_output, (float (*)[6][6][6])m->l_f->weight, m->l_f->d_preact);
	bp_preact_s1<<<64, 64>>>((float (*)[6][6])m->l_s1->d_preact, (float (*)[6][6])m->l_s1->d_output, (float (*)[6][6])m->l_s1->preact);
	bp_weight_s1<<<64, 64>>>((float (*)[4][4])m->l_s1->d_weight, (float (*)[6][6])m->l_s1->d_preact, (float (*)[24][24])m->l_c1->output);
	bp_bias_s1<<<64, 64>>>(m->l_s1->bias, (float (*)[6][6])m->l_s1->d_preact);

	bp_output_c1<<<64, 64>>>((float (*)[24][24])m->l_c1->d_output, (float (*)[4][4])m->l_s1->weight, (float (*)[6][6])m->l_s1->d_preact);
	bp_preact_c1<<<64, 64>>>((float (*)[24][24])m->l_c1->d_preact, (float (*)[24][24])m->l_c1->d_output, (float (*)[24][24])m->l_c1->preact);
	bp_weight_c1<<<64, 64>>>((float (*)[5][5])m->l_c1->d_weight, (float (*)[24][24])m->l_c1->d_preact, (float (*)[28])m->l_input->output);
	bp_bias_c1<<<64, 64>>>(m->l_c1->bias, (float (*)[24][24])m->l_c1->d_preact);


	apply_grad<<<64, 64>>>(m->l_f->weight, m->l_f->d_weight, m->l_f->M * m->l_f->N);
	apply_grad<<<64, 64>>>(m->l_s1->weight, m->l_s1->d_weight, m->l_s1->M * m->l_s1->N);
	apply_grad<<<64, 64>>>(m->l_c1->weight, m->l_c1->d_weight, m->l_c1->M * m->l_c1->N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn(Model *m)
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int iter = 1;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data, m);

			m->l_f->bp_clear();
			m->l_s1->bp_clear();
			m->l_c1->bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(m->l_f->d_preact, m->l_f->output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, m->l_f->d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_pass(m);
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28], Model *m)
{
	float res[10];

	forward_pass(data, m);

	unsigned int max = 0;

	cudaMemcpy(res, m->l_f->output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test(Model *m)
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data, m) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
