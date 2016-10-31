#include "Cluster.h"
#include "AM.h"
#include "Dependency.h"

extern void all_test();
extern void find_frequent_pattern();
extern int alg_cluster();

int main(int argc, char *argv[])
{
	string corpus_path = "index_corpus.index";
	int command;
	if (argc == 1)
	{
		cout << "No argument given. Exit" << endl;
		exit(0);
	}
	
	command = atoi(argv[1]);
	if (argc == 2)
	{
		cout << "No path. default path : " << corpus_path << endl;
	}
	else
	{
		corpus_path = argv[2];
	}


	//all_test();
	//alg_cluster();
	if (command == 1)
	{
		cluster_embedding();
	}
	else if (command == 2)
	{
		cout << "Task>> find_frequent_pattern " << endl;
		find_frequent_pattern(corpus_path);
	}
	else if (command == 3)
	{
		cout << "Task>> Resolve omission" << endl;
		resolve_ommission(corpus_path);
	}

	return 0;
}