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
#include <stdio.h>

#ifdef WINVS
using uint = size_t;
#else
#define uint size_t
#define scanf_s scanf
#define fscanf_s fscanf
#endif


using namespace std;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
static const std::string slash = "\\";
#else
static const std::string slash = "/";
#endif

static const string data_path = "data"+ slash;
static const string common_input = ".." + slash + ".." + slash +"input" + slash;

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
	sort(v.begin(), v.end(), less<T>());
}


template <typename T>
class Set2 : public set<T>
{
public:
	Set2() : set<T>(){}
	template<class _Iter>
	Set2(_Iter _First, _Iter _Last): set<T>(_First, _Last)
	{	
	}
	Set2(vector<T>& v) : set<T>(v.begin(), v.end())
	{
	}
	bool has(T elem){
		return (this->find(elem) != this->end());
	}
	void add(const vector<T>& v){
		for (const T& item : v){
			insert(item);
		}
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
void vector_add(vector<T>& first, const vector<T>& second)
{
	first.insert(first.end(), second.begin(), second.end());
}


template <typename T, typename U>
vector<U> parallelize(const vector<T>& input, function<U(T)> eval)
{
	int nThread = std::thread::hardware_concurrency();
	using ITR = typename vector<T>::const_iterator;

	function<vector<U>(ITR, ITR)> evaluator =
		[eval](ITR begin, ITR end)
	{
		vector<U> result;
		for (ITR itr = begin; itr != end; itr++)
		{
			result.push_back(eval(*itr));
		}
		return result;
	};

	vector<future<vector<U>>> f_list;
	uint unit = input.size() / nThread;
	for (uint i = 0; i < nThread; i++)
	{
		uint st = i * unit;
		uint ed = (i + 1) * unit;
		ITR itr_begin = input.begin() + st;
		ITR itr_end = input.begin() + ed;
		f_list.push_back(async(launch::async, evaluator, itr_begin, itr_end));
	}

	vector<U> merged;
	for (auto &f : f_list)
	{
		vector<U> temp = f.get();
		//vector_add(merged, temp);
	}
	return merged;
}

template <typename T>
class Counter : public map < T, int > {
public:
	void add_count(T& item)
	{
		if (this->find(item) == this->end())
		{
			(*this)[item] = 0;
		}
		(*this)[item] = (*this)[item] + 1;
	}
};

static void check_file(ifstream& infile, string& path)
{
	if (!infile.good())
	{
		cout << "File not valid : " + path << endl;
		exit(0);
	}
}


static vector<int> vector_and_(vector<int> v1, vector<int> v2)
{
	vector<int> result;
	// Two vector must be sorted
	auto itr1 = v1.begin();
	auto itr2 = v2.begin();
	while (itr1 != v1.end() && itr2 != v2.end())
	{
		if (*itr1 < *itr2)
			itr1++;
		else if (*itr1 > *itr2)
			itr2++;
		else if (*itr1 == *itr2)
		{
			result.push_back(*itr1);
			itr1++;
			itr2++;
		}
		else
			assert(false);
	}
	return result;
};
