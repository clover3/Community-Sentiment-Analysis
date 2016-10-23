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



bool sorted(vector<int> itemSet);
