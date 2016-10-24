#include "Dependency.h"
#include "Cluster.h"

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

Idx2Word* g_idx2word;

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
			// P(~item|remaining) = Count(~item & remaining) / Count(remaining)
			uint count_pattern = docs.count_occurence(pattern);
			uint count_remain = docs.count_occurence(remaining);
			float probability = float(count_pattern) / float(count_remain);

			uint count_remain_except = docs.count_occurence_except(remaining, item);
			float probability_not = float(count_remain_except) / float(count_remain);
			if (probability - probability_not > 0.2 && !ignore_pattern(pattern))
			{
				Dependency dep(item, remaining, probability);
				dependsList.push_back(dep);
				dep.print(*g_idx2word);
				cout << probability << " > " << probability_not << " ?? " << endl;
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

vector<Dependency> PatternToDependency(Docs& docs)
{
	vector<string> fslist = { "L2.txt", "L3.txt" }; // , "L4.txt", "L5.txt"};
	vector<Dependency> dependsList;
	for (string path : fslist)
	{
		FrequentSet fs(path);
		vector_add(dependsList, FS2Dependency(docs, fs));
	}

	return dependsList;
}

void apply_clustering(Doc& doc, map<int, int>& cluster)
{
	for (int& word : doc)
	{
		if (cluster.find(word) != cluster.end())
		{
			word = cluster[word] + 100000000;
		}
	}

}


Set2<int> FindOmission(
	Doc& doc,
	Doc& predoc,
	vector<Dependency>& dependencyList,
	map<int, string>& idx2word,
	map<int, int>& cluster)
{
	Set2<int> omission_symbol;

	Doc doc_this = doc;
	Doc doc_pre = predoc;
	apply_clustering(doc_this, cluster);
	apply_clustering(doc_pre, cluster);
	Set2<int> predoc_set(predoc);
	for (Dependency dependency : dependencyList)
	{
		sort(doc_this);
		if (contain(doc, dependency.dependents))
		{
			if (missing(doc, dependency.target) && predoc_set.has(dependency.target))
			{
				omission_symbol.insert(dependency.target);
			}
		}
	}

	Set2<int> omission_word;
	for (int word : doc_pre)
	{
		int symbol = word;
		if (cluster.find(word) != cluster.end())
			symbol = cluster[word];

		if (omission_symbol.has(symbol))
		{
			omission_word.insert(word);
		}
	}
	return omission_word;
}

// Doc 에서 생략을 찾자. 귀차느니까 바로 앞글을 dependency 라고 가정


void resolve_ommission()
{
	Docs docs("index_corpus.index");
	map<int, string> idx2word = load_idx2word("idx2word");
	g_idx2word = &idx2word;

	Docs indexdocs("index_corpus.index");

	map<int, int> cluster = loadCluster("cluster_1.txt");
	apply_clustering(indexdocs, cluster);
	apply_cluster(idx2word, cluster);
	vector<Dependency> dependsList = PatternToDependency(indexdocs);

	cout << "Loading raw sentence" << endl;
	ifstream fin("..\\..\\input\\bobae_raw_sentence.txt");
	vector<string> rawdoc;
	string temp;
	while (getline(fin, temp))
	{
		rawdoc.push_back(temp);
	}


	cout << "Now resolve omission" << endl;
	for (uint i = 1; i < docs.size(); i++)
	{
		set<int> ommision = FindOmission(docs[i], docs[i - 1], dependsList, idx2word, cluster);
		if (ommision.size() > 0)
		{
			cout << "--------------------------------------" << endl;
			cout << "Index = " << i << endl;
			cout << "prev : " << endl;
			print_doc(docs[i - 1], idx2word);
			cout << rawdoc[i - 1] << endl;
			cout << "this : " << endl;
			print_doc(docs[i], idx2word);
			cout << rawdoc[i] << endl;
			cout << "This sentence was missing : ";
			for (int token : ommision)
				cout << idx2word[token] << " ";
			cout << endl;
			scanf_s("%*c");
		}
	}
}

