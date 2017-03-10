#pragma once
#include "stdafx.h"

//------------ Type Definition -----------//


#include "ItemSet.h"
#include "vectorComp.h"
#include "Docs.h"
#include "FrequentSet.h"
#include "word2idx.h"

// ----------------------------------------//

Docs load_article(string path);

int count_occurence(const Docs& docs, ItemSet itemSet);

FrequentSet generate_candidate(FrequentSet L_k);
FrequentSet prune_candidate(const Docs& docs, const FrequentSet& C_k, const FrequentSet& L_prev, uint min_dup);

void print_function_complete(const char* function_name);

void find_frequent_pattern(string corpus_path);

void car_frequent_pattern(string corpus_path);