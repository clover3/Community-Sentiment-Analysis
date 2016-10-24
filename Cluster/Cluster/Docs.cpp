#include "AM.h"

void print_doc(Doc& doc, map<int,string>& idx2word)
{
	for (int word : doc)
	{
		cout << idx2word[word] << " ";
	}
	cout << endl;
}


void Docs::init(vector<Doc>& docs)
{
	for (int i = 0; i < docs.size(); i++)
	{
		push_back(docs[i]);
		for (int word : docs[i])
		{
			if (invIndex.find(word) == invIndex.end())
			{
				invIndex[word] = vector<int>();
			}
			invIndex[word].push_back(i);
		}
	}

	for (auto& key_value : invIndex)
	{
		sort(key_value.second);
	}
}

void Docs::rebuild_index()
{
	invIndex.clear();
	for (int i = 0; i < size(); i++)
	{
		for (int word : (*this)[i])
		{
			if (invIndex.find(word) == invIndex.end())
			{
				invIndex[word] = vector<int>();
			}
			invIndex[word].push_back(i);
		}
	}

	for (auto& key_value : invIndex)
	{
		sort(key_value.second);
	}
}

Docs::Docs(vector<Doc>& docs)
{
	init(docs);
}


Docs::Docs(string path)
{
	vector<Doc> rawDocs;
	ifstream infile(path);
	string line;
	while (std::getline(infile, line))
	{

		set<int> wordSet;
		std::istringstream iss(line);
		int token;
		while (!iss.eof()){
			iss >> token;
			wordSet.insert(token);
		}

		Doc doc(wordSet.begin(), wordSet.end());
		sort(doc.begin(), doc.end());
		rawDocs.push_back(doc);
	}
	init(rawDocs);
}

int Docs::max_word_index() const{
	int max = 5;
	for (auto doc : (*this)){
		for (int word : doc)
		{
			if (word > max && word < 1000000)
				max = word;
		}
	}
	return max;
}

void Docs::filter_not_in(Set2<int> interested_word)
{
	transform(begin(), end(), begin(), [&interested_word](Doc doc){
		Doc filtered_doc;
		for (int word : doc)
		{
			if (interested_word.has(word))
				filtered_doc.push_back(word);
		}
		return filtered_doc;
	});
}

size_t Docs::docsize() const
{
	size_t sum = 0;
	for (auto doc : (*this))
		sum += doc.size();

	return sum;
}

vector<int> Docs::get_occurence(int word) const{
	if (invIndex.find(word) == invIndex.end())
		return vector<int>();
	else
	{
		vector<int> v = invIndex.find(word).operator*().second;
		return v;
	}
}


uint Docs::count_occurence_single(int item) const
{
	return get_occurence(item).size();
}


function<vector<int>(vector<int>, vector<int>)> vector_and = [](vector<int> v1, vector<int> v2)
{
	vector<int> result;
	// Two vector must be sorted
	auto itr1 = v1.begin();
	auto itr2 = v2.begin();
	while (itr1 != v1.end() && itr2 != v2.end())
	{
		if (*itr1 < *itr2)
			itr1++;
		else if (*itr1 > *itr2)
			itr2++;
		else if (*itr1 == *itr2)
		{
			result.push_back(*itr1);
			itr1++;
			itr2++;
		}
		else
			assert(false);
	}
	return result;
};

int fun_global = 10;

function<vector<int>(vector<int>, vector<int>)> vector_except = [](vector<int> v1, vector<int> v2)
{
	vector<int> result;
	// Two vector must be sorted
	auto itr1 = v1.begin();
	auto itr2 = v2.begin();
	while (itr1 != v1.end() && itr2 != v2.end())
	{
		if (*itr1 < *itr2)
		{
			result.push_back(*itr1);
			itr1++;
		}
		else if (*itr1 > *itr2)
			itr2++;
		else if (*itr1 == *itr2)
		{
			itr1++;
		}
		else
			assert(false);
	}

	while (itr1 != v1.end())
	{
		result.push_back(*itr1);
		itr1++;
	}
	return result;
};

// TODO optimize it
uint Docs::count_occurence(ItemSet itemSet) const
{
	// Count using inverted index;
	vector<vector<int>> vector_occurrence;
	for (int item : itemSet){
		vector_occurrence.push_back(this->get_occurence(item));
	}

	vector<int> common_occurence = foldLeft(vector_occurrence, vector_occurrence[0], vector_and);
	return common_occurence.size();
}

uint Docs::count_occurence_except(ItemSet itemSet, int except) const
{
	// Count using inverted index;
	vector<vector<int>> vector_occurrence;
	for (int item : itemSet){
		vector_occurrence.push_back(this->get_occurence(item));
	}

	vector<int> common_occurence = foldLeft(vector_occurrence, vector_occurrence[0], vector_and);
	vector<int> except_occurence = get_occurence(except);

	vector<int> remain_occurence = vector_except(common_occurence, except_occurence);

	return remain_occurence.size();
}



void apply_clustering(Docs& docs, map<int, int>& cluster)
{

	int cluster_prefix = 100000000;
	if (docs.max_word_index() >= cluster_prefix)
		cluster_prefix = docs.max_word_index() + 1;


	ofstream fout("clustering.log");
	fout << cluster_prefix;
	fout.close();

	for (Doc &doc : docs)
	{
		for (int& word : doc)
		{
			if (cluster.find(word) != cluster.end())
			{
				word = cluster[word] + cluster_prefix;
			}
		}
	}
	docs.rebuild_index();
}