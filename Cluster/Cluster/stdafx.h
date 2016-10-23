#pragma once
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
#include <cctype>
#include <locale>

using uint = size_t;


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


template <typename T>
class Set2 : public set<T>
{
public:
	Set2() : set(){}
	template<class _Iter>
	Set2(_Iter _First, _Iter _Last): set(_First, _Last)
	{	
	}
	Set2(vector<T>& v) : set(v.begin(), v.end())
	{
	}
	bool has(T elem){
		return (this->find(elem) != this->end());
	}
};

// trim from start
static inline std::string &ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(),
		std::not1(std::ptr_fun<int, int>(std::isspace))));
	return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(),
		std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
	return ltrim(rtrim(s));
}

template <typename T>
void vector_add(vector<T>& first, vector<T>& second)
{
	first.insert(first.end(), second.begin(), second.end());
}