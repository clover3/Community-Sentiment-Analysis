#include "Affinity.h"


list<Affinity> EvalAffinity(FrequentSet fs, Docs& docs)
{
	list<Affinity> result;
	for (auto item : fs)
	{
		double a = affinity(item[0], item[1], docs);
		result.push_back(Affinity(item[0], item[1], a) );
	}
	return result;
}

void save_affinity(list<Affinity> &affinityList, string path)
{
	ofstream fout(path);
	for (auto a : affinityList)
	{
		fout << a.word1 << "\t" << a.word2 << "\t" << a.affinity << endl;
	}
	fout.close();
}

void save_affinity_viewable(list<Affinity> &affinityList, string path, MCluster& cluster, Idx2Word idx2word)
{
	auto idx2wordEx = FunctorIdx2Word(cluster, idx2word);
	ofstream fout(path);
	for (auto a : affinityList)
	{
		fout << idx2wordEx(a.word1) << "\t" 
			<< idx2wordEx(a.word2) << "\t" 
			<< a.affinity << endl;
	} 
	fout.close();
}


double affinity(const int itemA, const int itemB, const Docs& docs)
{
	vector<int> ab = vector<int>{itemA, itemB};
	uint count_a = docs.count_occurence_single(itemA);
	uint count_b = docs.count_occurence_single(itemB);
	uint count_ab = docs.count_occurence(ab);
	uint total = docs.size();

	double p_ab = float(count_ab) / total;
	double p_a = float(count_a) / total;
	double p_b = float(count_b) / total;
	double lift = p_ab / (p_a *p_b);
	return lift;
}

void affinity_job(string corpus_path)
{
	// Load clusters
	vector<string> cluster_path = { "cluster_4.txt", "cluster_6.txt", "cluster_car.txt" };
	MCluster mcluster;
	mcluster.add_clusters(cluster_path);

	// Load docs 
	Docs docs(corpus_path, mcluster);

	FrequentSet fs(data_path + "L_Dependency.txt");
	auto affinityList = EvalAffinity(fs, docs);
	save_affinity(affinityList, data_path + "CarAffinity.index");

	Idx2Word idx2word = load_idx2word(common_input + "idx2word");
	save_affinity_viewable(affinityList, data_path + "CarAffinity.txt", mcluster, idx2word);
}

