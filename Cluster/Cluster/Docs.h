#pragma once
#include "stdafx.h"
#include "word2idx.h"
#include "ItemSet.h"
#include "mcluster.h"

using Doc = vector <int>;

class IndexedDoc : public Doc
{
private:
	MCluster* m_pMCluster;
	Set2<int> categories;
	Set2<int> words;
public:
	IndexedDoc(const Doc& doc, MCluster& cluster) : vector<int>(doc), m_pMCluster(&cluster) 
	{  
		for (int word : doc)
		{
			this->categories.add(cluster.get_categories(word));
			this->words.insert(word);
		}
	}

	bool contains_category(Category_ID category)
	{
		return categories.has(category.get());
	}

	bool contains_word(Word_ID word)
	{
		return words.has(word.get());
	}

	Word_ID find_word_with_category(Category_ID category)
	{
		if (!this->contains_category(category))
			return Word_ID::Invalid();
		for (int item : *this)
		{
			vector<int> categories = m_pMCluster->get_categories(item);
			if (contain(categories, category.get()))
			{
				return Word_ID(item);
			}
		}
		return Word_ID::Invalid();
	}
};

void print_doc(ofstream& out, Doc& doc, map<int, string>& idx2word);
void print_doc(Doc& doc, map<int, string>& idx2word);

class Docs : public vector < Doc >
{
public:
	Docs(string path){};
	Docs(string path, MCluster& mcluster);
	Docs(vector<Doc>& docs);
	Docs(Idx2Word& idx2word, string path);

	size_t docsize() const;

	void rebuild_index();
	vector<int> get_occurence(int word) const;
	uint count_occurence_single(int item) const;
	uint count_occurence(vector<int> itemSet) const;
	uint count_occurence_except(ItemSet itemSet, int except) const;
	uint count_occurence_without(int except) const;
	uint count_occurence_without(int target, int except) const;
	int max_word_index() const;

	void filter_not_in(Set2<int> interested_word);
private:
	void init(vector<Doc>& docs);
	void init2(vector<Doc>& docs, MCluster& mcluster);
	std::map<int, vector<int>> invIndex;
};

void apply_clustering(Docs& docs, map<int, int>& cluster);

void save_docs(vector<Doc>& docs, string path);