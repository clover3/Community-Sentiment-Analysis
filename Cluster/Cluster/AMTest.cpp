#include "AM.h"
#include <cassert>


namespace tests
{
	void test_FrequentSet()
	{
		FrequentSet c1;
		ItemSet i1;
		i1.push_back(1); i1.push_back(2);

		ItemSet i2;
		i2.push_back(1); i2.push_back(2);


		c1.insert(i1);
		assert(c1.size() == 1);
		c1.insert(i2);
		assert(c1.size() == 1);

		print_function_complete(__FUNCTION__);
	}

	bool returnfalse(int dummy)
	{
		printf("check");
		return false;
	}

	void test_all_of()
	{
		vector<int> v = { 1, 2, 34, 5, 6, 7 };
		all_of(v.begin(), v.end(), returnfalse);

	}

	void test_joinable()
	{

		ItemSet i1;
		i1.push_back(1); i1.push_back(2); i1.push_back(4);

		ItemSet i2;
		i2.push_back(1); i2.push_back(2); i2.push_back(4);

		ItemSet i3;
		i3.push_back(1); i3.push_back(3); i3.push_back(4);

		ItemSet i4 = { 1, 2, 6 };

		assert(!ItemSet::joinable(i1, i2));
		assert(!ItemSet::joinable(i1, i3));
		assert(!ItemSet::joinable(i3, i4));
		assert(ItemSet::joinable(i1, i4));

		print_function_complete(__FUNCTION__);
	}



	void test_join()
	{

		ItemSet i1;
		i1.push_back(1); i1.push_back(2); i1.push_back(4);

		ItemSet i4;
		i4.push_back(1); i4.push_back(2); i4.push_back(6);

		ItemSet newset = ItemSet::join(i1, i4);

		assert(newset.size() == 4);
		assert(newset[0] == 1);
		assert(newset[1] == 2);
		assert(newset[2] == 4);
		assert(newset[3] == 6);

		print_function_complete(__FUNCTION__);
	}

	void test_all_true()
	{
		vector<bool> v1 = { true, false, true, false };
		vector<bool> v2 = { true, true, true, true };
		assert(!all_true(v1));
		assert(all_true(v2));

		print_function_complete(__FUNCTION__);
	}

	void test_subsets()
	{
		ItemSet i1 = { 1, 3, 5, 7 };

		auto sets = i1.subsets();

		assert(sets.size() == 4);

		assert(sets[0][0] == 3);
		assert(sets[0][1] == 5);
		assert(sets[0][2] == 7);

		assert(sets[1][0] == 1);
		assert(sets[1][1] == 5);
		assert(sets[1][2] == 7);

		print_function_complete(__FUNCTION__);
	}

	void all_test()
	{
		test_all_true();
		test_FrequentSet();
		test_join();
		test_joinable();
		test_subsets();
		test_all_of();
		print_function_complete(__FUNCTION__);
	}
}

void all_test()
{
	tests::all_test();
}