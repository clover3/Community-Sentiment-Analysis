#include "Dependency.h"
#include <math.h>
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


bool is_dependent(const int item, const ItemSet& pattern, const Docs& docs)
{
	ItemSet remaining = pattern - item;

	// Pattern = item + remaining
	uint count_pattern = docs.count_occurence(pattern);
	uint count_remain = docs.count_occurence(remaining);
	uint count_item = docs.count_occurence_single(item);
	float probability = float(count_pattern) / float(count_remain);
	// P(item|remaining) = Count(pattern) / Count(remaining)

	int remain_item = remaining[0];
	uint count_item_without_remain = docs.count_occurence_without(item, remain_item);
	uint count_without_remain = docs.count_occurence_without(remain_item);
	float probability_not = float(count_item_without_remain) / float(count_without_remain);
	// P(item|~remaining) = Count(item & ~remaining) / Count(~remaining)

	uint count_remain_except = docs.count_occurence_except(remaining, item);
	float probability_without = float(count_remain_except) / float(count_remain);
	// P(~item|remaining) = Count(~item & remaining) / Count(remaining)

	double lift = float(count_pattern) / sqrt(count_item *count_remain);

	if (remain_item == 100000883)
	{
		cout << (*g_idx2word)[item] << " | " << (*g_idx2word)[remain_item] << " lift = " << lift << " p = " << probability << " p_w = " << probability_without << endl;
	}

	if (lift > 0.6 ) // && probability - probability_without > 0.2)
	{
		return true;
	}
	else
		return false;
}

vector<Dependency> get_dependency_mt(Docs& docs, FrequentSet& fs)
{
	cout << "Calculating Dependencies" << endl;
	vector<ItemSet> patterns;
	patterns.reserve(fs.size());

	for (ItemSet pattern : fs)
	{
		patterns.push_back(pattern);
	}

	function<vector<Dependency>(ItemSet)> check = 
		[docs](ItemSet pattern)
	{
		vector<Dependency> dependsList;
		//  pick one item.
		for (int item : pattern)
		{
			for (int item : pattern)
			{
				ItemSet remaining = pattern - item;
				if (is_dependent(item, pattern, docs))
				{
					Dependency dep(item, remaining, 0);
					dependsList.push_back(dep);
					dep.print(*g_idx2word);
				}
			}
		}
		return dependsList;
	};
	
	vector<vector<Dependency>> r_2d = parallelize(patterns, check);

	vector<Dependency> dependsList; 
	for (auto v : r_2d){
		vector_add(dependsList, v);
	}

	return dependsList;
}


vector<Dependency> get_dependency(Docs& docs, FrequentSet& fs)
{
	cout << "Calculating Dependencies" << endl;
	vector<ItemSet> patterns;
	patterns.reserve(fs.size());

	for (ItemSet pattern : fs)
	{
		patterns.push_back(pattern);
	}

	vector<Dependency> dependsList;
	for (auto pattern : patterns)
	{
		//  pick one item.
		for (int item : pattern)
		{
			ItemSet remaining = pattern - item;
			if (is_dependent(item, pattern, docs))
			{
				Dependency dep(item, remaining, 0);
				dependsList.push_back(dep);
				dep.print(*g_idx2word);
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
	vector<string> fslist = { data_path+"L2.txt", data_path+"L3.txt" }; // , "L4.txt", "L5.txt"};
	vector<Dependency> dependsList;
	for (string path : fslist)
	{
		FrequentSet fs(path);
		vector_add(dependsList, get_dependency(docs, fs));
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

void save_dependency(string path, vector<Dependency>& dependencys)
{
	ofstream fout(path);
	for (Dependency dep : dependencys)
	{
		fout << dep.target;
		for (int item : dep.dependents)
			fout << " " << item;
		fout << endl;
	}
	fout.close();
}


vector<Dependency> load_dependency(string path)
{
	ifstream infile(path);
	check_file(infile, path);

	vector<Dependency> dlist;
	string line;
	while (getline(infile, line))
	{
		vector<int> itemset;
		std::istringstream iss(line);
		int item;
		iss >> item;

		int token;
		while (!iss.eof()){
			iss >> token;
			itemset.push_back(token);
		}
		Dependency dep(item, itemset, 0);
		dlist.push_back(dep);
	}
	return dlist;
}

vector<Dependency> eval_dependency(string corpus_path)
{
	Docs docs(corpus_path);
	map<int, string> idx2word = load_idx2word(common_input + "idx2word");
	g_idx2word = &idx2word;

	Docs indexdocs(corpus_path);

	map<int, int> cluster = loadCluster(data_path + "cluster_ep200.txt");
	apply_clustering(indexdocs, cluster);
	apply_cluster(idx2word, cluster);
	//vector<Dependency> dependsList = PatternToDependency(indexdocs);
	FrequentSet fs(data_path + "L2_ep200.txt");
	vector<Dependency> dependsList = get_dependency(indexdocs, fs);
	save_dependency(data_path + "dependency.index", dependsList);
	return dependsList;
}

void resolve_ommission(string corpus_path)
{
	cout << "Loading raw sentence" << endl;
	ifstream fin(common_input + "bobae_raw_sentence.txt");
	check_file(fin, common_input + "bobae_raw_sentence.txt");
	vector<string> rawdoc;
	string temp;
	while (getline(fin, temp))
	{
		rawdoc.push_back(temp);
	}

	MCluster mcluster;//TOOOOOOOOOOOOOOODOOOOOOOOOOOOOOOOOOOOOOO

	Docs docs(corpus_path, mcluster);

	map<int, string> idx2word = load_idx2word(common_input + "idx2word");
	vector<Dependency> dependsList = load_dependency(data_path + "dependency.index");
	map<int, int> cluster = loadCluster(data_path + "cluster_ep200.txt");
	cout << "Now resolve omission" << endl;
	for (uint i = 1; i < docs.size(); i++)
	{
		set<int> ommision = FindOmission(docs[i], docs[i - 1], dependsList, idx2word, cluster);
		if (ommision.size() > 0)
		{
			cout << "--------------------------------------" << endl;
			cout << "Index = " << i << endl;
			cout << "▶ prev\t: " << rawdoc[i - 1] << endl;
			cout << " tokens\t : [";
			print_doc(docs[i - 1], idx2word);
			cout << "]" << endl;
			cout << "▶ this\t:" << rawdoc[i] << endl;
			cout << " tokens\t : [";
			print_doc(docs[i], idx2word);
			cout << "]" << endl;

			cout << "This sentence was missing : ";
			for (int token : ommision)
				cout << idx2word[token] << " ";
			cout << endl;
			scanf_s("%*c");
		}
	}
}

