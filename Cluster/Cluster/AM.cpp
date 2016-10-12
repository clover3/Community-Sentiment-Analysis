#include "AM.h"


void print_function_complete(char* function_name)
{
	printf("%s Completed\n", function_name);
}

// TODO optimize it
int count_occurence(vector<Doc> docs, ItemSet itemSet)
{
	int count = 0;
	for (Doc doc : docs)
	{
		vector<bool> occur_check;
		occur_check.resize(itemSet.size(), false);
		for (int word : doc){
			for (int i = 0; i < itemSet.size(); i++)
			{
				if (word == itemSet[i])
					occur_check[i] = true;
			}
		}
		if (all_true(occur_check))
			count++;
	}
	return count;
}


bool joinable(ItemSet set1, ItemSet set2)
{
	assert(set1.size() == set2.size());
	bool suc = true;

	size_t ss = set1.size();

	for (unsigned i = 0; i < ss - 1; i++)
	{
		if (set1[i] != set2[i])
			return false;
	}

	if (set1[ss - 1] < set2[ss - 1])
		return true;
	else
		return false;
}

ItemSet join(const ItemSet set1, const ItemSet set2)
{
	// Assumed Joinable..
	size_t ss = set1.size();
	ItemSet newset(ss + 1);

	for (int i = 0; i < ss - 1; i++)
	{
		newset[i] = set1[i];
	}

	if (set1[ss-1] < set2[ss-1])
	{
		newset[ss - 1] = set1[ss - 1];
		newset[ss] = set2[ss - 1];
	}
	else
	{
		newset[ss - 1] = set2[ss - 1];
		newset[ss] = set1[ss - 1];
	}
	return newset;
}

bool all_true(vector<bool> v)
{
	return all_of(v.begin(), v.end(), [](bool f){return f; });
}

bool all_of(vector<ItemSet> iterable, function<bool(ItemSet&)> contain)
{
	return all_of(iterable.begin(), iterable.end(), contain);
}

function<bool(ItemSet&)> contain(const FrequentSet& fs)
{
	return [fs](ItemSet& itemSet){
		return (fs.find(itemSet) != fs.end());
	};
}

vector<ItemSet> subsets(ItemSet& itemSet)
{
	vector<ItemSet> result;
	result.reserve(itemSet.size());
	for (int i = 0; i < itemSet.size(); i++)
	{
		ItemSet newset;
		newset.reserve(itemSet.size());
		for (int j = 0; j < itemSet.size(); j++)
		{
			if (i != j)
				newset.push_back(itemSet[j]);
		}
		result.push_back(newset);
	}
	return result;
}

size_t item_size(FrequentSet fs)
{
	return fs.begin()->size();
}

bool sorted(vector<int> itemSet)
{
	int last = itemSet[0];
	for (int i = 1; i < itemSet.size(); i++)
	{
		if (itemSet[i] < last)
			return false;
		last = itemSet[i];
	}
	return true;
}

FrequentSet generate_candidate(FrequentSet L_k)
{
	FrequentSet C_next;
	for (ItemSet set1 : L_k)
	{
		assert(sorted(set1));
		for (ItemSet set2 : L_k)
		{
			if (joinable(set1, set2))
			{
				C_next.insert(join(set1, set2));
			}
		}
	}
	return C_next;
}

struct PMArgs{
	const vector<Doc>* docs;
	const FrequentSet* C_k;
	const FrequentSet* L_prev;
	int min_dup;
};

FrequentSet prune_candidate_v(
	vector<ItemSet>::iterator begin, 
	vector<ItemSet>::iterator end,
	const vector<Doc>& docs, 
	const FrequentSet& L_prev, 
	int min_dup
	)
{
	// Make L2
	FrequentSet L2;
	int prune = 0;
	int nprune = 0;
	for (vector<ItemSet>::iterator itr = begin ; itr != end ; itr++)
	{
		ItemSet candidate = *itr;
		// TODO Prue by L_prev
		if (all_of(subsets(candidate), contain(L_prev)))
		{
			nprune++;
			//count occurence
			int occurence = count_occurence(docs, candidate);
			if (occurence >= min_dup)
			{
				L2.insert(candidate);
			}
		}
		else
			prune++;
	}
	//printf("p/n = %d/%d\n", prune, nprune);
	
	return L2;
}


FrequentSet prune_candidate_mt(const vector<Doc>& docs, const FrequentSet& C_k, const FrequentSet& L_prev, int min_dup)
{
	int nThread = 8;
	vector<ItemSet> data(C_k.begin(), C_k.end());
	int interval = data.size() / nThread;
	vector<future<FrequentSet>> fVector;
	for (int i = 0; i < nThread; i++)
	{
		vector<ItemSet>::iterator begin, end;
		begin = data.begin() + interval * i;
		if (i + 1 < nThread)
			end = data.begin() + interval * (i + 1);
		else
			end = data.end();

		fVector.push_back(async(prune_candidate_v, begin, end, docs, L_prev, min_dup));
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

FrequentSet prune_candidate(const vector<Doc>& docs,const FrequentSet& C_k,const FrequentSet& L_prev, int min_dup)
{
	// Make L2
	vector<ItemSet> v(C_k.begin(), C_k.end());
	return prune_candidate_v(v.begin(), v.end(), docs, L_prev, min_dup);
}

DWORD lt = GetTickCount();
DWORD elapsed()
{
	DWORD dt = GetTickCount() - lt;
	lt = GetTickCount();
	return dt;
}

FrequentSet build_C2_sub(const vector<ItemSet>& l1, int st1, int ed1, int st2, int ed2)
{
	FrequentSet FS;
	for (int i = st1; i < ed1; i++)
	{
		for (int j = max(st2,st1+1); j < ed2; j++)
		{
			int lastItem1 = l1[i][0];
			int lastItem2 = l1[j][0];
			if (lastItem1 < lastItem2)
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

bool comp(ItemSet& i1, ItemSet& i2)
{
	if (i1.size() != i2.size())
		return i1.size() < i2.size();

	for (int i = 0; i < i1.size(); i++)
	{
		if (i1[i] != i2[i])
			return i1[i] < i2[i];
	}
	return false;
}

void sort(vector<ItemSet>& v)
{
	sort(v.begin(), v.end(), comp);
}

FrequentSet build_C2(FrequentSet L1)
{
	FrequentSet C2;
	int middle = L1.size() / 2;
	FrequentSet c_sub[4];

	vector<ItemSet> l1_vector = vector<ItemSet>(L1.begin(), L1.end());
	sort(l1_vector);
	int size = L1.size();
	C2 = build_C2_sub(l1_vector, 0, size, 0, size);

	for (ItemSet set1 : L1)
	{
		for (ItemSet set2 : L1)
		{
			int lastItem1 = set1[0];
			int lastItem2 = set2[0];
			if (lastItem1 < lastItem2)
			{
				ItemSet newset;
				newset.push_back(lastItem1);
				newset.push_back(lastItem2);
				C2.insert(newset);
			}
		}
	}
	return C2;
}

void am(vector<Doc> docs, int max_word_id)
{
	vector<int> counts;
	counts.resize(max_word_id+1, 0);
	int min_dup = 100;
	
	for (vector<int> doc : docs){
		for (int word : doc){
			counts[word] = counts[word] + 1;
		}
	}

	vector<FrequentSet> L;
	vector<FrequentSet> C;
	L.resize(10);
	C.resize(10);

	cout << "Making\t L1..." ;

	for (int i = 0; i < max_word_id; i++)
	{
		if (counts[i] >= min_dup)
		{
			vector<int> items;
			items.push_back(i);
			L[0].insert(items);
		}
	}
	cout << L[0].size() << " sets. " << elapsed() << "ms" << endl;
	cout << "Generate C2...";
	C[1] = build_C2(L[0]);
	cout << C[1].size() << " sets. " << elapsed() << "ms" <<endl;

	cout << "Pruning L2...";
	// Make L2
	L[1] = prune_candidate_mt(docs, C[1], L[0], min_dup);
	cout << L[1].size() << " sets. " << elapsed() << "ms" << endl;

	// Generate C3

	for (int i = 2; i < 10; i++)
	{
		cout << "Generate C" << i+1 << "...";
		C[i] = generate_candidate(L[i-1]);
		cout << C[i].size() << " sets. " << elapsed() << "ms" << endl;
		// Make L3

		cout << "Pruning L" << i + 1 << "...";
		L[i] = prune_candidate_mt(docs, C[i], L[i-1], min_dup);
		cout << L[i].size() << " sets. " << elapsed() << "ms" << endl;
	}

	
}


vector<Doc> load_article(string path)
{
	vector<Doc> result;

	ifstream infile(path);
	string line;
	while (std::getline(infile, line))
	//for (int i = 0; i < 10000; i++)
	{
		std::getline(infile, line);
		
		set<int> wordSet;
		std::istringstream iss(line);
		int token;
		while (!iss.eof()){
			iss >> token;
			wordSet.insert(token);
		}

		Doc doc(wordSet.begin(), wordSet.end());
		sort(doc.begin(), doc.end());
		result.push_back(doc);
	}

	return result;
}

int max_word_index(Corpus docs){
	int max = 5;
	for (auto doc : docs){
		for (int word : doc)
		{
			if (word > max)
				max = word;
		}
	}
	printf("max :%d\n", max);
	return max;
}

void am_main()
{
	vector<Doc> docs = load_article("bobae_as_index");
	am(docs, max_word_index(docs));

}