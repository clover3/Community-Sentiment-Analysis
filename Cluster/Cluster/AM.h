#pragma once
#include "stdafx.h"

//------------ Type Definition -----------//


#include "ItemSet.h"
#include "vectorComp.h"
#include "Docs.h"
#include "FrequentSet.h"
#include "word2idx.h"

// ----------------------------------------//

Docs load_article(string path);

int count_occurence(const Docs& docs, ItemSet itemSet);

FrequentSet generate_candidate(FrequentSet L_k);
FrequentSet prune_candidate(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, uint min_dup);

void print_function_complete(const char* function_name);

template <typename T>
class Counter : public map < T, int > {
public:
	void add_count(T& item)
	{
		if (this->find(item) == this->end())
		{
			(*this)[item] = 0;
		}
		(*this)[item] = (*this)[item] + 1;
	}
};

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

set<int> FindOmission(Doc& doc, vector<Dependency>& dependencyList, Doc& predoc, map<int, string>& idx2word);