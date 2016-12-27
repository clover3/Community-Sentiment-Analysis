#include "AM.h"
#include "timeAux.h"
#include "Cluster.h"
#include "mcluster.h"

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

vector<ItemSet> generate_candidate_sub(
	const vector<ItemSet>& l1, 
	uint st1, uint ed1, uint st2, uint ed2, 
	const MCluster& mcluster)
{
	vector<ItemSet> FS;
	for (uint i = st1; i < ed1; i++)
	{
		for (uint j = max(st2, st1 + 1); j < ed2; j++)
		{
			ItemSet set1 = l1[i];
			ItemSet set2 = l1[j];
			assert(sorted(set1));
			assert(sorted(set2));
			if (ItemSet::joinable(set1, set2, mcluster))
			{
				FS.push_back(ItemSet::join(set1, set2));
			}
		}
	}
	return FS;
}

FrequentSet generate_candidate_mt(FrequentSet L_k, const MCluster& mcluster)
{
	FrequentSet C_next;
	vector<ItemSet> vect(L_k.begin(), L_k.end());
	using pairint = pair<int, int>;
	int split = 8;
	int size = (int)vect.size();
	int interval = size / split;
	Set2<pairint> intervals;
	for (int i = 0; i < split + 1; i++)
	{
		int start = i * interval;
		int end = min((i+1) *interval, size);
		intervals.insert(pair<int,int>(start, end));
	}

	vector<pair<pairint, pairint>> inputs = combination(intervals, intervals);

	function<vector<ItemSet>(pair<pairint, pairint>)> worker = [mcluster, vect](pair<pairint, pairint> interval_pair){
		int st1 = interval_pair.first.first;
		int ed1 = interval_pair.first.second;
		int st2 = interval_pair.second.first;
		int ed2 = interval_pair.second.second;
		return generate_candidate_sub(vect, st1, ed1, st2, ed2, mcluster);
	};

	vector<vector<ItemSet>> results = parallelize(inputs, worker);

	for (auto v : results){
		for (ItemSet item : v)
		{
			C_next.insert(item);
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
		C[i] = generate_candidate_mt(L[i-1], mcluster);
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
    vector<string> cluster_path = {"cluster_0.txt", "cluster_1.txt", "cluster_2.txt", "cluster_3.txt", "cluster_4.txt",
                                "cluster_5.txt", "cluster_6.txt", "cluster_7.txt", "cluster_8.txt", "cluster_9.txt"};
    MCluster mcluster;
	mcluster.add_clusters(cluster_path);

	Docs docs(corpus_path, mcluster);
	ExtractFrequent(docs, mcluster);
}

