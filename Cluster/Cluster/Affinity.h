#include "stdafx.h"
#include "AM.h"
#include "Cluster.h"

class Affinity{
public:

	int word1;
	int word2;
	double affinity;
	Affinity(int w1, int w2, double a) : word1(w1), word2(w2), affinity(a)
	{}
};

double affinity(const int itemA, const int itemB, const Docs& docs);
void affinity_job(string corpus_path);