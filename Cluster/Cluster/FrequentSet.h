#pragma once
#include "stdafx.h"
#include "ItemSet.h"

class FrequentSet : public set < ItemSet >
{

public:
	FrequentSet(string path);
	FrequentSet() : set(){};
	using set::set;
	void save(string path) const;
	uint item_size() const;
};