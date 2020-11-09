all:
	nvcc -O0 -g -std=c++14 -lcuda -lcublas *.cu -o CNN

run:
	./CNN
clean:
	rm CNN
