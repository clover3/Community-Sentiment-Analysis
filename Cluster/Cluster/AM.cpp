#include "AM.h"
#include "timeAux.h"
#include "Cluster.h"

void print_function_complete(const char* function_name)
{
	printf("%s Completed\n", function_name);
}

FrequentSet generate_candidate(FrequentSet L_k, const MCluster& mcluster)
{
	FrequentSet C_next;
	for (ItemSet set1 : L_k)
	{
		assert(sorted(set1));
		for (ItemSet set2 : L_k)
		{
			if (ItemSet::joinable(set1, set2, mcluster))
			{
				C_next.insert(ItemSet::join(set1, set2));
			}
		}
	}
	return C_next;
}

FrequentSet prune_candidate_v(
	vector<ItemSet>::iterator begin, 
	vector<ItemSet>::iterator end,
	const Docs& docs,
	const FrequentSet& L_prev, 
	uint min_dup
	)
{
	// Make L2
	FrequentSet L2;
	for (vector<ItemSet>::iterator itr = begin ; itr != end ; itr++)
	{
		ItemSet candidate = *itr;
		//count occurence
		uint occurence = docs.count_occurence(candidate);
		if (occurence >= min_dup)
		{
			L2.insert(candidate);
		}
	}
	
	return L2;
}


FrequentSet prune_candidate_mt(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, uint min_dup)
{
	int nThread = std::thread::hardware_concurrency();
    printf("Working on %d threads\n", nThread);
	assert(nThread >= 0 );
	assert(nThread < 256);

	vector<ItemSet> data(C_k.begin(), C_k.end());
	uint interval = data.size() / nThread;
	vector<future<FrequentSet>> fVector;
	for (int i = 0; i < nThread; i++)
	{
		vector<ItemSet>::iterator begin, end;
		begin = data.begin() + interval * i;
		if (i + 1 < nThread)
			end = data.begin() + interval * (i + 1);
		else
			end = data.end();

		fVector.push_back(async(launch::async, prune_candidate_v, begin, end, docs, L_prev, min_dup));
	}

	
	FrequentSet fs_all;
	for (future<FrequentSet> &f : fVector)
	{
		FrequentSet fs = f.get();
		for (ItemSet item : fs)
		{
			fs_all.insert(item);
		}
	}
	return fs_all;
}

FrequentSet prune_candidate(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, uint min_dup)
{
	// Make L2
	vector<ItemSet> v(C_k.begin(), C_k.end());
	return prune_candidate_v(v.begin(), v.end(), docs, L_prev, min_dup);
}


FrequentSet build_C2_sub(const vector<ItemSet>& l1, uint st1, uint ed1, uint st2, uint ed2, const MCluster& mcluster)
{
	FrequentSet FS;
	for (uint i = st1; i < ed1; i++)
	{
		for (uint j = max(st2,st1+1); j < ed2; j++)
		{
			int lastItem1 = l1[i][0];
			int lastItem2 = l1[j][0];
			if (lastItem1 < lastItem2 && mcluster.different(lastItem1, lastItem2))
			{
				ItemSet newset;
				newset.push_back(lastItem1);
				newset.push_back(lastItem2);
				FS.insert(newset);
			}
		}
	}
	return FS;
}

void sort(vector<ItemSet>& v)
{
	sort(v.begin(), v.end(), ItemSet::comp);
}

FrequentSet build_C2(FrequentSet L1, const MCluster& mcluster)
{
	FrequentSet C2;
	uint middle = L1.size() / 2;
	FrequentSet c_sub[4];

	vector<ItemSet> l1_vector = vector<ItemSet>(L1.begin(), L1.end());
	sort(l1_vector);
	uint size = L1.size();
	C2 = build_C2_sub(l1_vector, 0, size, 0, size, mcluster);
	/*
	for (ItemSet set1 : L1)
	{
		for (ItemSet set2 : L1)
		{
			int lastItem1 = set1[0];
			int lastItem2 = set2[0];
			if (lastItem1 < lastItem2 && mcluster.different(lastItem1, lastItem2))
			{
				ItemSet newset;
				newset.push_back(lastItem1);

				newset.push_back(lastItem2);
				C2.insert(newset);
			}
		}
	}*/

	return C2;
}



string genLpath(int i)
{
	return data_path + "L" + to_string(i) + ".txt";
}

void ExtractFrequent(Docs& docs, MCluster& mcluster)
{
	uint min_dup = 20;


	vector<FrequentSet> L;
	vector<FrequentSet> C;
	L.resize(10);
	C.resize(10);


	cout << "Making\t L1..." ;

	vector<int> all_words = mcluster.get_all_words();
	for (int word : all_words)
	{
		uint count = docs.count_occurence_single(word);
		if (count >= min_dup)
		{
			ItemSet items;
			items.push_back(word);
			L[0].insert(items);
		}
	}


	vector<int> all_categories = mcluster.get_all_categorys();
	for (int word : all_categories)
	{
		uint count = docs.count_occurence_single(word);
		if (count >= min_dup)
		{
			ItemSet items;
			items.push_back(word);
			L[0].insert(items);
		}
	}

	cout << L[0].size() << " sets. " << elapsed() << "ms" << endl;
	L[0].save(genLpath(1));

//	printf("Doc size reduction : %d->", docs.docsize());
//	docs.filter_not_in(L1);
	printf("%d\n", docs.docsize());

	cout << "Generate C2...";
	C[1] = build_C2(L[0], mcluster);
	cout << C[1].size() << " sets. " << elapsed() << "ms" <<endl;

	cout << "Pruning L2...";
	// Make L2
	L[1] = prune_candidate_mt(docs, C[1], L[0], min_dup);
	cout << L[1].size() << " sets. " << elapsed() << "ms" << endl;
	L[1].save(genLpath(2));
	// Generate C3

	for (int i = 2; i < 10; i++)
	{
		cout << "Generate C" << i+1 << "...";
		C[i] = generate_candidate(L[i-1], mcluster);
		cout << C[i].size() << " sets. " << elapsed() << "ms" << endl;
		// Make L3

		cout << "Pruning L" << i + 1 << "...";
		L[i] = prune_candidate_mt(docs, C[i], L[i - 1], min_dup);
		cout << L[i].size() << " sets. " << elapsed() << "ms" << endl;
		L[i].save(genLpath(i+1));

		if (L[i].size() == 0)
			break;
	}
}

void find_frequent_pattern(string corpus_path)
{
	cout << "Loading clusters...";
	map<int, int> cluster1 = loadCluster(data_path + "cluster_ep20.txt");
	map<int, int> cluster2 = loadCluster(data_path + "cluster_ep200.txt");
	MCluster mcluster;
	mcluster.add_cluster(cluster1, 10000000);
	mcluster.add_cluster(cluster2, 20000000);
	cout << endl;

	Docs docs(corpus_path, mcluster);
	ExtractFrequent(docs, mcluster);
}

