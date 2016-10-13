#include <vector>
#include <cassert>
#include <set>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>

#include <algorithm>
#include <future>
#include <Windows.h>
using namespace std;

//------------ Type Definition -----------//

using Doc = vector <int>;
using ItemSet = vector <int> ;
using FrequentSet = set<ItemSet>;
using Corpus = vector < Doc > ;

// ----------------------------------------//

vector<Doc> load_article(string path);
bool all_true(vector<bool> v);
size_t item_size(FrequentSet fs);

vector<ItemSet> subsets(ItemSet& itemSet);
bool joinable(ItemSet set1, ItemSet set2);
ItemSet join(const ItemSet set1, const ItemSet set2);


FrequentSet generate_candidate(FrequentSet L_k);
FrequentSet prune_candidate(const vector<Doc>& docs, const FrequentSet& C_k, const FrequentSet& L_prev, int min_dup);


void print_function_complete(char* function_name);

template <typename T>
class Set2 : public set<T>
{
public:
	bool has(T elem){
		return (this->find(elem) != this->end());
	}
};