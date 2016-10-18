#include "stdafx.h"

//------------ Type Definition -----------//

using Doc = vector <int>;
using ItemSet = vector <int> ;
using FrequentSet = set<ItemSet>;
class Docs : public vector < Doc >
{
public:
	Docs(vector<Doc>& docs);
	vector<int> get_occurence(int word) const{
		if (invIndex.find(word) == invIndex.end())
			return vector<int>();
		else
		{
			vector<int> v = invIndex.find(word).operator*().second;
			return v;
		}
	}
private:
	std::map<int, vector<int>> invIndex;
};


template <typename T>
class Set2 : public set<T>
{
public:
	bool has(T elem){
		return (this->find(elem) != this->end());
	}
};
// ----------------------------------------//

vector<Doc> load_article(string path);
bool all_true(vector<bool> v);
size_t item_size(FrequentSet fs);

vector<ItemSet> subsets(ItemSet& itemSet);
bool joinable(ItemSet set1, ItemSet set2);
ItemSet join(const ItemSet set1, const ItemSet set2);


FrequentSet generate_candidate(FrequentSet L_k);
FrequentSet prune_candidate(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, int min_dup);

void save_FrequentSet(string,const FrequentSet&);
void print_function_complete(const char* function_name);

