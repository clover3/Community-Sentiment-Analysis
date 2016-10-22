#pragma once
#include "stdafx.h"
#include "word2idx.h"
#include "ItemSet.h"

using Doc = vector <int>;

void print_doc(Doc& doc, map<int, string>& idx2word);

class Docs : public vector < Doc >
{
public:
	Docs(string path);
	Docs(vector<Doc>& docs);
	Docs(Idx2Word& idx2word, string path);

	size_t docsize() const;

	vector<int> get_occurence(int word) const;
	uint count_occurence_single(int item) const;
	uint count_occurence(ItemSet itemSet) const;
	int max_word_index() const;

	void filter_not_in(Set2<int> interested_word);
private:
	void init(vector<Doc>& docs);
	std::map<int, vector<int>> invIndex;
};
 