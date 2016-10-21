#pragma once
#include "vectorComp.h"

bool all_true(vector<bool> v)
{
	return all_of(v.begin(), v.end(), [](bool f){return f; });
}

bool sorted(vector<int> itemSet)
{
	int last = itemSet[0];
	for (int i = 1; i < itemSet.size(); i++)
	{
		if (itemSet[i] < last)
			return false;
		last = itemSet[i];
	}
	return true;
}