CC=gcc
CXX=g++
RM=rm
CFLAGS=-std=c++11 -Wall -Wsign-compare -O3 
CPPFLAGS=-g $(shell root-config --cflags)
LDFLAGS=-g $(shell root-config --ldflags)
LDLIBS= -lpthread

SRCS=AM.cpp  AMTest.cpp  Cluster.cpp  Docs.cpp  FrequentSet.cpp  ItemSet.cpp  main.cpp  vectorComp.cpp  word2idx.cpp
OBJS=$(subst .cc,.o,$(SRCS))

all: Cluster 

Cluster: $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o Cluster $(OBJS) $(LDLIBS) 

depend: .depend

.depend: $(SRCS)	
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

dist-clean: clean
	$(RM) *~ .depend

include .depend
