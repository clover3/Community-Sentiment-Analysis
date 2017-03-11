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

string join(vector<string> v, string delimiter)
{
	string result = "";
	if (v.size() > 0)
		result += v[0];
	for (int i = 1; i < v.size(); i++)
		result += delimiter + v[i];
	return result;
}