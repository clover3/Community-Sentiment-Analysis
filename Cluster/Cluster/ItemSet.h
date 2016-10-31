#pragma once
#include "stdafx.h"

template< typename T, size_t N >
std::vector<T> makeVector(const T(&data)[N])
{
	return std::vector<T>(data, data + N);
}


class ItemSet : public vector <int>
{
public:
	ItemSet(size_t size) : vector(size) {};
	ItemSet() : vector(){};
	ItemSet(initializer_list<int> il) : vector(il) {};

	using vector::vector;

	vector<ItemSet> subsets() const;

	static bool joinable(ItemSet set1, ItemSet set2);
	static ItemSet join(const ItemSet set1, const ItemSet set2);
	static bool comp(ItemSet& i1, ItemSet& i2);
	ItemSet operator-(int item) const;
};