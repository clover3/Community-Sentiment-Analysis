#include "AM.h"


void FrequentSet::save(string path) const
{
	ofstream out(path);
	for (ItemSet item : (*this))
	{
		for (int token : item)
			out << token << "\t";
		out << endl;
	}
}

size_t FrequentSet::item_size() const
{
	return begin()->size();
}


function<bool(ItemSet&)> contain(const FrequentSet& fs)
{
	return [fs](ItemSet& itemSet){
		return (fs.find(itemSet) != fs.end());
	};
}