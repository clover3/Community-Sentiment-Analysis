#pragma once
#include "stdafx.h"


bool all_true(vector<bool> v);



template <typename T>
bool all_of(vector<T> iterable, function<bool(T&)> contain)
{
	return all_of(iterable.begin(), iterable.end(), contain);
}


bool sorted(vector<int> itemSet);