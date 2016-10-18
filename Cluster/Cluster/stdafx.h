
#undef NDEBUG

#include <algorithm>
#include <cassert>
#include <future>
#include <functional>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <sstream>
#include <stdio.h>
#include <set>
#include <string>
#include <vector>

#ifdef WINVS
#include <Windows.h>
#endif


using namespace std;


template <typename T, typename U>
U foldLeft(const std::vector<T>& data,
	const U& initialValue,
	const std::function<U(U, T)>& foldFn) {
	typedef typename std::vector<T>::const_iterator Iterator;
	U accumulator = initialValue;
	Iterator end = data.cend();
	for (Iterator it = data.cbegin(); it != end; ++it) {
		accumulator = foldFn(accumulator, *it);
	}
	return accumulator;
}

template <typename T>
std::vector<T> filter(const std::vector<T>& data, std::function<bool(T)> filterFn) {
	std::vector<T> result;
	foldLeft<T, std::vector<T>&>(data, result, [filterFn](std::vector<T>& res, T value)  -> std::vector<T>& {
		if (filterFn(value)) {
			res.push_back(value);
		}
		return res;
	});
	return result;
}


template <typename T, typename U>
std::vector<U> mapf(const std::vector<T>& data, const std::function<U(T)> mapper) {
	std::vector<U> result;
	foldLeft<T, std::vector<U>&>(data, result, [mapper](std::vector<U>& res, T value)  -> std::vector<U>& {
		res.push_back(mapper(value));
		return res;
	});
	return result;
}

template<class T> 
void sort(vector<T>& v)
{	
	sort(v.begin(), v.end(), less<>());
}


