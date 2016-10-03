#include <iostream>
#include <fstream>
#include <stdio.h>
#include <list>
#include <vector>
#include <set>
#include <string>
#include <map>
using namespace std;

class Embedding : public vector < float >
{
public:
	string text;
};
using Embeddings = vector<Embedding> ;

Embeddings* loadEmbeddings(char* path)
{
	Embeddings *ptr = new Embeddings;
	ifstream fin(path);
	int nEntry, nDim;
	string str;
	fin >> nEntry >> nDim;
	nEntry = 10000;
	printf("Entry :%d \nDimension : %d\n", nEntry, nDim); 

	for (int i = 0; i < nEntry; i++)
	{
		Embedding v;
		fin >> v.text;
		for (int j = 0; j < nDim; j++)
		{
			float f;
			fin >> f;
			v.push_back(f);
		}
		ptr->push_back(v);
	}
	return ptr;
}


class Labels : public vector<int> 
{
public:
	Labels(int size)
	{
		this->resize(size);
		for (unsigned int i = 0; i < this->size(); i++)
		{
			this->operator[](i) = i;
		}
	}
};

class Cluster
{
public:
	static Labels thresholdCluster(Embeddings* eb, float eps);
};

float dist(const Embedding &e1, const Embedding &e2)
{
	float acc = 0; 
	for (unsigned int i = 0; i < e1.size(); i++)
	{
		float d = (e1[i] - e2[i]);
		acc += d * d;
	}
	return acc;
}

class Edges : public vector < list<int> > {
};




Labels Cluster::thresholdCluster(Embeddings* eb, float eps)
{
	printf("thresholdCluster ENTRY\n");
	bool retry = true;
	int nNode = eb->size();
	Edges edges;
	edges.resize(nNode);
	printf("bulding edges...\n");
	// build edges;
	int nEdge = 0;
	float eps2 = eps*eps;
	try{
		for (unsigned int i = 0; i < eb->size(); i++)
		{
			cout << i << " ";
			for (unsigned int j = i + 1; j < eb->size(); j++)
			{
				float d = dist((*eb)[i], (*eb)[j]);
				if (d < eps2)
				{
					// add edge
					edges[i].push_back(j);
					edges[j].push_back(i);
					nEdge += 2;
				}
			}
		}
	}
	catch (...)
	{

	}
	printf("built %d edges\n", nEdge);

	// init labels
	Labels labels(nNode);

	printf("running clustering...\n");

	vector<int> rank(nNode);
	for (auto r : rank)
		r = 0;

	while (retry)
	{
		retry = false;
		for (unsigned int i = 0; i < nNode; i++)
		{	
			// spread my label
			list<int>::iterator itr;
			for (auto target : edges[i])
			{
				if (labels[i] != labels[target])
				{
					if (rank[labels[i] ] > rank[labels[target]])
					{
						labels[target] = labels[i];
					}
					else if (rank[labels[i]] == rank[labels[target]])
					{
						labels[target] = labels[i];
						rank[labels[i]] = rank[labels[i]] + 1;
					}
					else
						labels[i] = labels[target];

					retry = true;
				}
			}
		}
	}

	return labels;
}

void display(Labels& label, Embeddings* eb)
{
	set<int> distinctLabel(label.begin(), label.end());
	map<int, vector<int>> group;

	for (unsigned int i = 0; i < eb->size(); i++)
	{
		int l = label[i];
		group[l].push_back(i);
	}

	for (auto & v : group)
	{
		if (v.second.size() > 1)
		{
			cout << v.first << " ";
			for (auto item : v.second)
			{
				cout << (*eb)[item].text << " ";
			}
			printf("\n");
		}
	}

	printf("Number of label : %d\n", distinctLabel.size());

}

void output(Labels& label, Embeddings* eb)
{
	set<int> distinctLabel(label.begin(), label.end());
	map<int, vector<int>> group;

	for (unsigned int i = 0; i < eb->size(); i++)
	{
		int l = label[i];
		group[l].push_back(i);
	}

	ofstream fout("output.txt");
	for (auto & v : group)
	{
		if (v.second.size() > 1)
		{
			fout << v.first ;
			for (auto item : v.second)
			{
				fout << "," << item ;
			}
			fout << endl;
		}
	}

	fout.close();
}

void runner()
{
	printf("Runner ENTRY\n");
	char path[] = "C:\\work\\Code\\Community-Sentiment-Analysis\\input\\korean_word2vec_wv_50.txt";
	Embeddings* eb = loadEmbeddings(path);

	Labels label = Cluster::thresholdCluster(eb, 10);

	display(label, eb);

	delete eb;
}

int main()
{
	runner();
	return 0;
}