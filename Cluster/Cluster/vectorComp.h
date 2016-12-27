#pragma once
#include "stdafx.h"


bool all_true(vector<bool> v);



template <typename T>
bool all_of(vector<T> iterable, function<bool(T&)> contain)
{
	return all_of(iterable.begin(), iterable.end(), contain);
}


template <typename T>
bool contain(vector<T> big_set, vector<T> sub_set)
{
	auto itr1 = big_set.begin();
	auto itr2 = sub_set.begin();
	int match = 0;
	while (itr1 != big_set.end() && itr2 != sub_set.end())
	{
		if (*itr1 < *itr2)
			itr1++;
		else if (*itr1 > *itr2)
			itr2++;
		else if (*itr1 == *itr2)
		{
			match++;
			itr1++;
			itr2++;
		}
		else
			assert(false);
	}
	return match == sub_set.size();
}

template <typename T>
bool contain(vector<T> big_set, T elem)
{
	for (T item : big_set)
	{
		if (item == elem)
			return true;
	}
	return false;
}

static int max(vector<int>& v)
{
	int max_val = INT_MIN;
	for (int elem : v)
	{
		if (elem > max_val)
			max_val = elem;
	}
	return max_val;
}


bool sorted(vector<int> itemSet);

template <typename T>
vector<pair<T, T>> combination(Set2<T> s1, Set2<T> s2)
{
	vector<pair<T, T>> plist;
	for (auto item : s1){
		for (auto item2 : s2)
		{
			plist.push_back(pair<T, T>(item, item2));
		}
	}
	return plist;
}