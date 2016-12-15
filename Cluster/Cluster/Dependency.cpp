#include "Dependency.h"
#include <math.h>
#include "Cluster.h"
#include <fstream>

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
function<string(int)> lambda_idx2word = [](int idx){ return string("NULL");  };

bool is_dependent(const int item, const ItemSet& pattern, const Docs& docs)
{
	ItemSet remaining = pattern - item;

	uint total = docs.size();
	// Pattern = item + remaining
	uint count_pattern = docs.count_occurence(pattern);
	uint count_remain = docs.count_occurence(remaining);
	uint count_item = docs.count_occurence_single(item);
	/*
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
	 */
	double p_ir = float(count_pattern) / total;
	double p_i = float(count_item) / total;
	double p_r = float(count_remain) / total;

	double lift = p_ir / (p_i *p_r);

	if (lift > 3 )//  probability - probability_without > 0.2)
	//if (probability / probability_without > 2)
	{
		//cout << lift << " " << p_ir << " " << p_i << " " << p_r << endl;
		return true;
	}
	else
		return false;
}

void log_is_dependent(const int item, const ItemSet& pattern, const Docs& docs, function<string(int)> toStr, ofstream& out)
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

	out << toStr(item) << "|" << toStr(remain_item) << endl
		<< " Lift=" << lift << endl
		<< "P(item|remain)=" << probability << endl
		<< "P(item|~remaining)=" << probability_not << endl
		<< "P(~item|remaining)=" << probability_without << endl;
}

vector<Dependency> get_dependency(Docs& docs, MCluster& mcluster, FrequentSet& fs)
{
	cout << "Calculating Dependencies" << endl;
	vector<ItemSet> patterns;
	patterns.reserve(fs.size());

	for (ItemSet pattern : fs)
	{
		patterns.push_back(pattern);
	}

	ofstream fstream("depend.log");

	vector<Dependency> dependsList;
	for (auto pattern : patterns)
	{
		//  pick one item.
		for (int item : pattern)
		{
			ItemSet remaining = pattern - item;
			//log_is_dependent(item, pattern, docs, lambda_idx2word, fstream);
			if (is_dependent(item, pattern, docs))
			{
				Dependency dep(item, remaining, 0);
				dependsList.push_back(dep);
				dep.print(lambda_idx2word);
			}
		}
	}

	return dependsList;
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
			ItemSet remaining = pattern - item;
			if (is_dependent(item, pattern, docs))
			{
				Dependency dep(item, remaining, 0);
				dependsList.push_back(dep);
				//dep.print(lambda_idx2word);
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


// token can be both(real_word, category)
bool contain_cword(Doc& doc, int token, MCluster& mcluster)
{
	for (auto item : doc)
	{
		if (item == token) // Case real word
			return true;

		// case category 
		vector<int> categories = mcluster.get_categories(item);
		if (contain(categories, token))
			return true;
	}
	return false;
}

bool missing_cword(Doc& doc, int token, MCluster& mcluster)
{
	return !contain_cword(doc, token, mcluster);
}

bool contain_cwords(Doc& doc, vector<int> cwords, MCluster& mcluster)
{
	for (auto target : cwords)
	{
		if (missing_cword(doc, target, mcluster))
			return false;
	}
	return true;
}

vector<Dependency> PatternToDependency(Docs& docs, MCluster& mcluster)
{
	vector<string> fslist = { data_path+"L2.txt", data_path+"L3.txt" }; // , "L4.txt", "L5.txt"};
	vector<Dependency> dependsList;
	for (string path : fslist)
	{
		FrequentSet fs(path);
		vector_add(dependsList, get_dependency(docs, mcluster, fs));
	}

	return dependsList;
}


Set2<int> FindOmission(
	Doc& doc,
	Doc& predoc,
	vector<Dependency>& dependencyList,
	map<int, string>& idx2word,
	MCluster& mcluster)
{
	Set2<int> omission_symbol;

	for (Dependency dependency : dependencyList)
	{
		if (contain_cwords(doc, dependency.dependents, mcluster))
		{
			if (missing_cword(doc, dependency.target, mcluster)
				&& contain_cword(predoc, dependency.target, mcluster) )
			{
				omission_symbol.insert(dependency.target);
			}
		}
	}

	Set2<int> possible_omission_words;
	for (int cword : omission_symbol)
	{
		possible_omission_words.insert(cword);
		possible_omission_words.add(mcluster.get_words(cword));
	}

	Set2<int> omission_word;
	for (int word : predoc)
	{
		if (possible_omission_words.has(word))
		{
			omission_word.insert(word);
		}
	}
	return omission_word;
}

// Doc ���� ������ ã��. �������ϱ� �ٷ� �ձ��� dependency ��� ����

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
	map<int, string> idx2word = load_idx2word(common_input + "idx2word");
	g_idx2word = &idx2word;

	cout << "Loading clusters...";
	vector<string> cluster_path = { "cluster_0.txt", "cluster_1.txt", "cluster_2.txt", "cluster_3.txt", "cluster_4.txt",
		                             "cluster_5.txt", "cluster_6.txt", "cluster_7.txt", "cluster_8.txt", "cluster_9.txt" };

	MCluster mcluster;
	mcluster.add_clusters(cluster_path);

	lambda_idx2word = [mcluster, idx2word](int idx){
		if (idx > TEN_MILLION)
		{
			vector<int> idxs = mcluster.get_words(idx);
			size_t sublen = min(idxs.size(), (size_t)10);
			vector<int> subwords(idxs.begin(), idxs.begin() + sublen);
			function<string(int)> mapper = [idx2word](const int idx){
				auto token = idx2word.find(idx);
				if (token != idx2word.end()){
					string word = token.operator*().second;
					return word;
				}
				else
					return string("null"); 
			};
			vector<string> words = mapf(subwords, mapper);
			string ret = "g[";
			for (auto word : words)
				ret += string(word + ",");
			ret += "]";
			return ret;
		}
		else
		{
			string ret = idx2word.find(idx).operator*().second;
			return ret;
		}
	};

	cout << "done" << endl;

	Docs docs(corpus_path, mcluster);
	FrequentSet fs(data_path + "L2.txt");
	vector<Dependency> dependsList = get_dependency_mt(docs, fs);
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

	vector<string> cluster_path = { "cluster_0.txt", "cluster_1.txt", "cluster_2.txt", "cluster_3.txt", "cluster_4.txt" };
	MCluster mcluster;
	mcluster.add_clusters(cluster_path);

	Docs docs(corpus_path, mcluster);

	map<int, string> idx2word = load_idx2word(common_input + "idx2word");
	vector<Dependency> dependsList = load_dependency(data_path + "dependency.index");
	cout << "Now resolve omission" << endl;
	for (uint i = 9000; i < docs.size(); i++)
	{
		set<int> ommision = FindOmission(docs[i], docs[i - 1], dependsList, idx2word, mcluster);
		if (ommision.size() > 0)
		{
			cout << "--------------------------------------" << endl;
			cout << "Index = " << i << endl;
			cout << "�� prev\t: " << rawdoc[i - 1] << endl;
			cout << " tokens\t : [";
			print_doc(docs[i - 1], idx2word);
			cout << "]" << endl;
			cout << "�� this\t:" << rawdoc[i] << endl;
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

