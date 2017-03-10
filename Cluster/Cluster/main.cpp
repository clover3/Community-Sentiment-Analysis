#include "Cluster.h"
#include "AM.h"
#include "Dependency.h"

extern void all_test();
extern void find_frequent_pattern();
extern int alg_cluster();
extern void resolve_omission_indexed();
extern void affinity_job(string corpus_path);

int main(int argc, char *argv[])
{
	string corpus_path = "index_corpus.index";
	int command;
	if (argc == 1)
	{
		cout << "No argument given." << endl;
		cout << "---- Usage -----" << endl;
		cout << " ./cluster [command] [corpus_path] " << endl;
		cout << " Commands " << endl;
		cout << "  1 : run cluster_embedding. Embedding path is hard coded " << endl;
		cout << "  2 : run find_frequent_pattern. " << endl;
		cout << "  3 : run eval_dependency. " << endl;
		cout << "  4 : run resolve_ommission. " << endl;
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
		cout << "Task>> Eval Dependency" << endl;
		eval_dependency(corpus_path);
	}
	else if (command == 4)
	{
		cout << "Task>> Resolve omission" << endl;
		resolve_ommission(corpus_path);
	}
	else if (command == 5)
	{
		cout << "Task>> Recover omission" << endl;
		resolve_omission_indexed();
	}
	else if (command == 6)
	{
		cout << "Task>> car_frequent_pattern " << endl;
		car_frequent_pattern(corpus_path);
	}
	else if (command == 7)
	{
		cout << "Task>> Eval Effinity" << endl;
		affinity_job(corpus_path);
	}
	return 0;
}