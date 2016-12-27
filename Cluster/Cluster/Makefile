CC=gcc
CXX=g++
RM=rm
CFLAGS=-std=c++11 -Wall -Wno-sign-compare -O3 
CPPFLAGS=-g $(shell root-config --cflags)
LDFLAGS=-g $(shell root-config --ldflags)
LDLIBS= -lpthread

SRCS=AM.cpp  AMTest.cpp  Cluster.cpp Dependency.cpp Docs.cpp  embedding.cpp FrequentSet.cpp  ItemSet.cpp  main.cpp  vectorComp.cpp  word2idx.cpp  mcluster.cpp
OBJS=$(subst .cc,.o,$(SRCS))
DELOBJS=$(subst .o,$(SRCS))

all: Cluster 

Cluster: $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o Cluster $(OBJS) $(LDLIBS) 

depend: .depend

.depend: $(SRCS)	
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(DELOBJS)

dist-clean: clean
	$(RM) *~ .depend

include .depend
