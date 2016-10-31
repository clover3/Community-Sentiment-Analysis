#pragma once
#include "AM.h"

class Dependency
{
public:
	int target;
	vector<int> dependents;
	float probability;
	Dependency(int t, vector<int> ds, float p) : target(t), dependents(ds), probability(p)
	{

	}
	void print(Idx2Word& idx2word)
	{
		cout << "P(" << idx2word[target] << " | ";
		for (auto ritem : dependents)
			cout << idx2word[ritem] << ", ";
		cout << ") = " << probability << endl;
	}
private:

};


void find_frequent_pattern();



Set2<int> FindOmission(
	Doc& doc,
	Doc& predoc,
	vector<Dependency>& dependencyList,
	map<int, string>& idx2word,
	map<int, int>& cluster);


void resolve_ommission(string corpus_path);