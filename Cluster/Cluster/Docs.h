#pragma once
#include "stdafx.h"
#include "ItemSet.h"

using Doc = vector <int>;
class Docs : public vector < Doc >
{
public:
	Docs(string path);
	Docs(vector<Doc>& docs);
	vector<int> get_occurence(int word) const;
	size_t docsize() const;
	uint count_occurence(ItemSet itemSet) const;
	void filter_not_in(Set2<int> interested_word);
	int max_word_index() const;
private:
	void init(vector<Doc>& docs);
	std::map<int, vector<int>> invIndex;
};
