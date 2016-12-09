#pragma once
#include "stdafx.h"
#include "word2idx.h"

template<typename T>
void for_enum(T& container, std::function<void(int, typename T::value_type&)> op)
{
	int idx = 0;
	for (auto& value : container)
		op(idx++, value);
}

class Embedding : public vector < float >
{
public:
	string text;
};
using Embeddings = vector<Embedding>;

class Labels : public vector<int>
{
public:
	Labels(size_t size)
	{
		this->resize(size);
		for (unsigned int i = 0; i < this->size(); i++)
		{
			this->operator[](i) = i;
		}
	}
};

enum EDIST_METRIC{
	euclidean = 1,
	manhattan = 2,
	geomean = 3

};

float dist_euclidean(const vector<float> &e1, const  vector<float> &e2);

class Centroids;
class Centroid : public vector < float > {
public:
	void begin_add()
	{
		add_cnt = 0;
	}
	Centroid(uint k) : vector<float>() 
	{
		for (uint i = 0; i < k; i++)
			push_back(0);
		begin_add();
	}
	void operator=(vector<float>& v)
	{
		for (uint i = 0; i < size(); i++)
			(*this)[i] = v[i];
	}
	void operator+=(Embedding& eb)
	{
		assert(add_cnt >= 0);
		assert(this->size() == eb.size());
		for (int i = 0; i < size(); i++)
		{
			(*this)[i] += eb[i];
		}
		add_cnt++;
	}

	void comlete_add()
	{
		for (float& v : (*this))
		{
			v /= add_cnt;
		}
		add_cnt = 0;
	}
private:
	int add_cnt = 0;
};

class Centroids : public vector<Centroid> {
public:
	Centroids(uint k, size_t dim)
	{
		for (uint i = 0; i < k; i++)
			this->push_back(Centroid(dim));
	}
};

class Clustering
{
public:
	static Labels thresholdCluster(Embeddings* eb, float eps);
	static Labels OneStepCluster(Embeddings* eb, float eps);
	static Labels KMeans(Embeddings* eb, float eps, int k);
	static Labels KMeans(Embeddings* eb, Centroids centroids, float eps, int k);
	static vector<Labels> Hierarchial(Embeddings* eb, vector<float> eps);
private:
};

using cluster = map < int, vector<int> > ;



class Edges : public vector < list<int> > {
public:
	Edges(Embeddings* eb, float eps, EDIST_METRIC dist_metric);
	int totalEdge(){ return nSize;  }
private:


	int nSize;
};

int find_min(const vector<float>& source, const vector<Centroid>& candidates);
void cluster_embedding();

map<int, int> loadCluster(string path);
void save_cluster(string path, Embeddings& eb, Word2Idx& word2idx, Labels& labels);

void apply_cluster(Idx2Word& idx2word, map<int, int>& cluster);

class MCluster
{
public:
	vector<int> get_categories(int word) const;
	vector<int> get_words(int category) const;

	bool different(int cword1, int cword2) const;
	void add_cluster(map<int, int>& cluster, int prefix);

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
};