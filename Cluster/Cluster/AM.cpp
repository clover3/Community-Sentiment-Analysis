#include "AM.h"
#include "timeAux.h"


void print_function_complete(const char* function_name)
{
	printf("%s Completed\n", function_name);
}

FrequentSet generate_candidate(FrequentSet L_k)
{
	FrequentSet C_next;
	for (ItemSet set1 : L_k)
	{
		assert(sorted(set1));
		for (ItemSet set2 : L_k)
		{
			if (ItemSet::joinable(set1, set2))
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
	int prune = 0;
	int nprune = 0;
	for (vector<ItemSet>::iterator itr = begin ; itr != end ; itr++)
	{
		ItemSet candidate = *itr;
		// TODO Prue by L_prev
		bool fCountRequired = true; // (candidate.size() == 2 || all_of(subsets(candidate), contain(L_prev)));
		if (fCountRequired)
		{
			nprune++;
			//count occurence
			uint occurence = docs.count_occurence(candidate);
			if (occurence >= min_dup)
			{
				L2.insert(candidate);
			}
		}
		else
			prune++;
	}
	
	return L2;
}


FrequentSet prune_candidate_mt(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, int min_dup)
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

FrequentSet prune_candidate(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, int min_dup)
{
	// Make L2
	vector<ItemSet> v(C_k.begin(), C_k.end());
	return prune_candidate_v(v.begin(), v.end(), docs, L_prev, min_dup);
}


FrequentSet build_C2_sub(const vector<ItemSet>& l1, uint st1, uint ed1, uint st2, uint ed2)
{
	FrequentSet FS;
	for (uint i = st1; i < ed1; i++)
	{
		for (uint j = max(st2,st1+1); j < ed2; j++)
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

void sort(vector<ItemSet>& v)
{
	sort(v.begin(), v.end(), ItemSet::comp);
}

FrequentSet build_C2(FrequentSet L1)
{
	FrequentSet C2;
	uint middle = L1.size() / 2;
	FrequentSet c_sub[4];

	vector<ItemSet> l1_vector = vector<ItemSet>(L1.begin(), L1.end());
	sort(l1_vector);
	uint size = L1.size();
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



string genLpath(int i)
{
	return string("L") + to_string(i) + string(".txt");
}

void ExtractFrequent(Docs& docs, int max_word_id)
{
	vector<int> counts;
	counts.resize(max_word_id+1, 0);
	int min_dup = 100;
	
	for (vector<int> doc : docs){
		for (int word : doc){
			counts[word] = counts[word] + 1;
		}
	}

	Set2<int> L1;
	vector<FrequentSet> L;
	vector<FrequentSet> C;
	L.resize(10);
	C.resize(10);


	cout << "Making\t L1..." ;

	for (int i = 0; i < max_word_id; i++)
	{
		if (counts[i] >= min_dup)
		{
			ItemSet items;
			items.push_back(i);
			L[0].insert(items);
			L1.insert(i);
		}
	}
	cout << L[0].size() << " sets. " << elapsed() << "ms" << endl;
	L[0].save(genLpath(1));

	printf("Doc size reduction : %d->", docs.docsize());
	docs.filter_not_in(L1);
	printf("%d\n", docs.docsize());

	cout << "Generate C2...";
	C[1] = build_C2(L[0]);
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
		C[i] = generate_candidate(L[i-1]);
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

map<int, string> load_idx2word(string path)
{
	map<int, string> idx2word;
	ifstream infile(path);
	string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		int idx;
		iss >> idx;
		string word;
		getline(iss, word);
		idx2word[idx] = word.substr(1);
	}
	return idx2word;
}

bool ignore_pattern(ItemSet itemSet)
{
	for (int item : itemSet)
	{
		if (item == 5 || item == 491 || item == 72738 || item == 40405 || item == 248 || item == 31509)
			return true;
	}
	return false;
}

int find_match(int i, vector<string>& rawdoc, Doc doc, Idx2Word& idx2word)
{
	int run = 0;
	for (int j = i; run < 1000; j++)
	{
		run++;
		int cnt = 0;
		for (int word : doc)
		{
			if (rawdoc[j].find(idx2word[word]) != string::npos)
			{
				cnt++;
			}
		}
		if (float(cnt) >= float(doc.size()) * 0.5)
			return j;
	}
	return -1;
}

vector<Dependency> FS2Dependency(Docs& docs, FrequentSet& fs)
{
	vector<Dependency> dependsList;

	cout << "Calculating Dependencies" << endl;
	for (ItemSet pattern : fs)
	{
		//  pick one item.
		for (int item : pattern)
		{
			ItemSet remaining = pattern - item;

			// Pattern = item + remaining
			// P(item|remaining) = Count(pattern) / Count(remaining)
			int count_pattern = docs.count_occurence(pattern);
			int count_remain = docs.count_occurence(remaining);
			float probability = float(count_pattern) / float(count_remain);
			if (probability > 0.5 && !ignore_pattern(pattern))
			{
				Dependency dep(item, remaining, probability);
				dependsList.push_back(dep);
			}
		}
	}
	return dependsList;
}



bool missing(Doc& doc, int token)
{
	for (auto item : doc)
	{
		if (item == token)
			return false;
	}
	return true;
}

void PatternToDependency(Docs& docs)
{
	FrequentSet fs("L5.txt");
	map<int, string> idx2word = load_idx2word("idx2word");

	vector<Dependency> dependsList = FS2Dependency(docs, fs);
	//vector<Dependency> dependsList = FS2Dependency(docs, fs);

	cout << "Loading raw sentence" << endl;
	ifstream fin("..\\..\\input\\bobae_raw_sentence.txt");
	vector<string> rawdoc;
	string temp;
	while ( getline(fin, temp) )
	{ 
		rawdoc.push_back(temp);
	}


	cout << "Now resolve omission" << endl;
	for (uint i = 1; i < docs.size() ;i++)
	{
		set<int> ommision = FindOmission(docs[i], dependsList, docs[i - 1], idx2word);
		if (ommision.size() > 0)
		{
			cout << "--------------------------------------" << endl;
			cout << "Index = " << i << endl;
			cout << "prev : " << endl;
			print_doc(docs[i - 1], idx2word);
			cout << "this : " << endl;
			print_doc(docs[i], idx2word);
			cout << "This sentence was missing : ";
			for (int token : ommision)
				cout << idx2word[token] << " ";
			cout << endl;
			scanf_s("%*c");
		}
	}
}

set<int> FindOmission(Doc& doc, vector<Dependency>& dependencyList, Doc& predoc, map<int, string>& idx2word)
{
	set<int> omission;
	Set2<int> predoc_set(predoc.begin(), predoc.end());
	//Set2<int> doc_set(doc.begin(), doc.end());
	for (Dependency dependency : dependencyList)
	{
		sort(doc);
		if (contain(doc, dependency.dependents))
		{
			if (missing(doc, dependency.target) && predoc_set.has(dependency.target))
			{
				omission.insert(dependency.target);
			}
		}
	}
	return omission;
}

// Doc 에서 생략을 찾자. 귀차느니까 바로 앞글을 dependency 라고 가정






void am_main()
{
	Docs docs("input");
	//ExtractFrequent(docs, docs.max_word_index());
	PatternToDependency(docs);
}
