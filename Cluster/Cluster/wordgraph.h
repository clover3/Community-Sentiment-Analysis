#include "Dependency.h"
#pragma once

class WordGraph
{
public:
	WordGraph(size_t maxWord, vector<Dependency> dependList, MCluster& cluster)
	{
		all_edges.resize(maxWord + 1);
		for (auto d : dependList)
		{
			Set2<Word_ID> word_a = expand_word(d.target, cluster);
			Set2<Word_ID> word_b = expand_word(d.dependents[0], cluster);

			for(auto pair_ab : combination(word_a,word_b))
			{
				int index = pair_ab.first.get();
				all_edges[index].insert(pair_ab.second.get());
			}
		}
	}

	set<int> neighbor(Word_ID source, Doc& doc)
	{
		set<int> nlist;
		assert(source.valid());
		Set2<int> edges = all_edges[source.get()];
		for (int item : doc)
		{
			if (edges.has(item))
				nlist.insert(item);
		}
		return nlist;
	}

	bool no_neighbor(Word_ID source, Doc& doc)
	{
		assert(source.valid());
		Set2<int> edges = all_edges[source.get()];
		for (int item : doc)
		{
			if (edges.has(item))
				return false;
		}
		return true;
	}
	bool has_neighbor(Word_ID source, Doc& doc){ return !no_neighbor(source, doc); }
	set<int> word_context(Word_ID source, Doc& doc, Doc& context)
	{
		if (neighbor(source, doc).size() > 2)
			return set<int>();
		return neighbor(source, context);
	}
private:
	vector<Set2<int>> all_edges;
};
