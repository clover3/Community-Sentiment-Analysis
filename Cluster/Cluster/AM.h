#pragma once
#include "stdafx.h"

//------------ Type Definition -----------//


#include "ItemSet.h"
#include "vectorComp.h"
#include "Docs.h"

// --------- class Definition --------------//
class FrequentSet : public set < ItemSet >
{

public:
	using set::set;
	void save(string path) const;
	uint item_size() const;
};



// ----------------------------------------//

Docs load_article(string path);

int count_occurence(const Docs& docs, ItemSet itemSet);

FrequentSet generate_candidate(FrequentSet L_k);
FrequentSet prune_candidate(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, uint min_dup);

void print_function_complete(const char* function_name);

