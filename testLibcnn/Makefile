CC := g++
CFLAGS := -O3 -g -Wall -Wextra -Werror -std=c++14 
# CFLAGS := -O0 -g -Wall -Wextra -Werror -std=c++14
LIBS:=-L../libfaascnn/ -lfaascnn -lpthread
INCLUDES:=-I../libfaascnn

test: main.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) 

%.o: %.cpp
	$(CC) -c $< $(CFLAGS) $(INCLUDES)

clean:
	rm -f *.o test 
