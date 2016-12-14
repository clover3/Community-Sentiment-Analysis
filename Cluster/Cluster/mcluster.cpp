#include "mcluster.h"

vector<int> MCluster::get_categories(int word) const
{
	if (word2categories.find(word) == word2categories.end())
	{
		return vector<int>();
	}

	vector<int> v = word2categories.find(word).operator*().second;
	return v;
}

vector<int> MCluster::get_words(int category) const
{
	if (category2words.find(category) == category2words.end())
	{
		return vector<int>();
	}

	vector<int> v = category2words.find(category).operator*().second;
	return v;
}

bool MCluster::different(int cword1, int cword2) const
{

	vector<int> v1, v2;
	if (cword1 > TEN_MILLION)
		v1 = get_words(cword1);
	else
		v1.push_back(cword1);

	if (cword2 > TEN_MILLION)
		v2 = get_words(cword2);
	else
		v2.push_back(cword2);

	vector<int> vr = vector_and_(v1, v2);
	return vr.size() == 0;
}

void MCluster::add_cluster(map<int, int>& cluster, int prefix)
{
	for (auto item : cluster)
	{
		int voca = item.first;
		int category = prefix + item.second;

		if (word2categories.find(voca) == word2categories.end())
			word2categories[voca] = vector<int>();

		word2categories[voca].push_back(category);


		if (category2words.find(category) == category2words.end())
			category2words[category] = vector<int>();

		category2words[category].push_back(voca);
	}
}

void MCluster::add_clusters(vector<string> paths)
{
	for (auto path : paths)
	{
		map<int, int> cluster1 = loadCluster(data_path + path);
		this->add_cluster(cluster1, m_base_id);
		m_base_id += TEN_MILLION;
	}
}
