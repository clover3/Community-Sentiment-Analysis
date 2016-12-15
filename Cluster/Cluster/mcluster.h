#pragma once
#include "Cluster.h"

/*
  This class handles multiple clusters and 
  enable indexing/referencing by the cluster's grouping id
*/

const int TEN_MILLION = 10000000;
const int MILLION = 1000000;

class Word_ID
{
private:
	int val;
public:
	static Word_ID Invalid(){ return Word_ID(-1); }
	bool valid(){ return val > 0; }
	int get(){ return val; }
	Word_ID(int id) : val(id){
		assert(id >= -1);
	}
};

class Category_ID
{
private:
	int val;
public:
	bool valid(){ return val > TEN_MILLION; }
	int get(){ return val; }
	Category_ID(int id) : val(id){
		assert(id >= -1);
	}
};

class MCluster
{
public:
	MCluster(){ m_base_id = TEN_MILLION; }
	vector<int> get_categories(int word) const;
	vector<int> get_words(int category) const;

	bool different(int cword1, int cword2) const;
	void add_cluster(map<int, int>& cluster, int prefix);
	void add_clusters(vector<string> paths);

	vector<int> get_all_words() const
	{
		std::vector<int> words;
		for (auto pair : word2categories)
			words.push_back(pair.first);
		return words;
	}
	vector<int> get_all_categorys() const
	{
		std::vector<int> v;
		for (auto pair : category2words)
			v.push_back(pair.first);
		return v;
	}
private:
	map<int, vector<int>> word2categories;
	map<int, vector<int>> category2words;
	int m_base_id;

};
