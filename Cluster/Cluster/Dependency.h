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
	void print(function<string(int)> idx2word)
	{
		cout << "P(" << idx2word(target) << " | ";
		for (auto ritem : dependents)
			cout << idx2word(ritem) << ", ";
		cout << ") = " << probability << endl;
	}
private:

};


class DependencyIndex
{
private:
	MCluster *m_pMCluster;
	vector<vector<Dependency>> index;
public:

	DependencyIndex(vector<Dependency>& vd, MCluster* mcluster)
	{
		cout << "DependencyIndex constructor" << endl;
		m_pMCluster = mcluster;
		int max_voca = max(mcluster->get_all_words());
		index.resize(max_voca+1);
		for (Dependency&d : vd)
		{
			if (d.dependents.size() == 0)
				cout << "Dammn!!" << endl;
			int dependents = d.dependents[0];
			if (dependents > TEN_MILLION)
			{
				vector<int> group = mcluster->get_words(dependents);
				for (int word : group)
					index[word].push_back(d);
			}
			else
			{
				index[dependents].push_back(d);
			}
		}
	}

	vector<Dependency> find_with_dependent(Word_ID dependent) const
	{
		assert(200000 > dependent.get());
		if (dependent.get() >= index.size())
			return vector<Dependency>();
		vector<Dependency> list1 = index[dependent.get()];
		return list1;
	}
};


void find_frequent_pattern();



Set2<int> FindOmission(
	Doc& doc,
	Doc& predoc,
	vector<Dependency>& dependencyList,
	map<int, string>& idx2word,
	map<int, int>& cluster);


vector<Dependency> eval_dependency(string corpus_path);
void resolve_ommission(string corpus_path);