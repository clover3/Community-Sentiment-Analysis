# I am a comment, and I want to say that the variable CC will be
# the compiler to use.
CC=g++
# Hey!, I am comment number 2. I want to say that CFLAGS will be the
# options I'll pass to the compiler.
CFLAGS=-std=c++11 -c -Wall -O3 
LDLIBS  = -pthread
PROG = Cluster


all: $(PROG) 

$(PROG): main.o AM.o AMTest.o
	$(CC) main.o AM.o AMTest.o -o $@ $(LDLIBS) -Wl,--no-as-needed

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

AM.o: AM.cpp
	$(CC) $(CFLAGS) AM.cpp

AMTest.o: AMTest.cpp
	$(CC) $(CFLAGS) AMTest.cpp

clean:
	rm *.o Cluster 




