#include "ItemSet.h"

bool ItemSet::joinable(ItemSet set1, ItemSet set2)
{
	assert(set1.size() == set2.size());
	bool suc = true;

	size_t ss = set1.size();

	for (unsigned i = 0; i < ss - 1; i++)
	{
		if (set1[i] != set2[i])
			return false;
	}

	if (set1[ss - 1] < set2[ss - 1])
		return true;
	else
		return false;
}

ItemSet ItemSet::join(const ItemSet set1, const ItemSet set2)
{
	// Assumed Joinable..
	size_t ss = set1.size();
	ItemSet newset(ss + 1);

	for (int i = 0; i < ss - 1; i++)
	{
		newset[i] = set1[i];
	}

	if (set1[ss - 1] < set2[ss - 1])
	{
		newset[ss - 1] = set1[ss - 1];
		newset[ss] = set2[ss - 1];
	}
	else
	{
		newset[ss - 1] = set2[ss - 1];
		newset[ss] = set1[ss - 1];
	}
	return newset;
}


vector<ItemSet> ItemSet::subsets() const
{
	vector<ItemSet> result;
	result.reserve(this->size());
	for (int i = 0; i < this->size(); i++)
	{
		ItemSet newset;
		newset.reserve(this->size());
		for (int j = 0; j < this->size(); j++)
		{
			if (i != j)
				newset.push_back( (*this)[j] );
		}
		result.push_back(newset);
	}
	return result;
}


bool ItemSet::comp(ItemSet& i1, ItemSet& i2)
{
	if (i1.size() != i2.size())
		return i1.size() < i2.size();

	for (int i = 0; i < i1.size(); i++)
	{
		if (i1[i] != i2[i])
			return i1[i] < i2[i];
	}
	return false;
}


ItemSet ItemSet::operator-(int target) const
{
	ItemSet result;
	for (int item : *this)
	{
		if (item != target)
			result.push_back(item);
	}
	return result;
}