#include "AM.h"
#include "FrequentSet.h"

FrequentSet::FrequentSet(string path)
{
	ifstream infile(path);
	check_file(infile, path);

	string line;
	while (std::getline(infile, line))
	{
		ItemSet itemSet;
		std::istringstream iss(trim(line));
		int token;
		while (!iss.eof()){
			iss >> token;
			itemSet.push_back(token);
		}
		this->insert(itemSet);
	}
}


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

