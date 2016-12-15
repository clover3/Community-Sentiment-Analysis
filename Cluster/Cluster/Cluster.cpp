#include "stdafx.h"
#include "Cluster.h"
#include "ThreadPool.h"
#include "embedding.h"

int find_min(const vector<float>& source, const vector<Centroid>& candidates)
{
	int index_min = 0;
	float min_dist = dist_euclidean(source, candidates[0]);

	for (int i = 0; i < candidates.size(); i++)
	{
		const vector<float> &candidate = candidates[i];
		float dist = dist_euclidean(source, candidate);
		if (dist < min_dist)
		{
			index_min = i;
			min_dist = dist;
		}
	}
	return index_min;
}


bool closethan(float d, float eps, EDIST_METRIC metric)
{
	if (metric == euclidean)
	{
		return d < eps *  eps;
	}
	else if (metric == manhattan)
	{
		return d < eps;
	}
	else if (metric == geomean)
	{
		return d < eps;
	}
	else
		exit(1);
}

Edges::Edges(Embeddings* eb, float eps, EDIST_METRIC dist_metric = euclidean)
{
	this->nSize = 0;
	this->resize(eb->size());
	printf("bulding edges...\n");
	// build edges;
	function<float(Embedding, Embedding)> distFunc;

	if (dist_metric == euclidean)
	{
		distFunc = dist_euclidean;
	}
	else if (dist_metric == manhattan)
	{
		distFunc = dist_manhattan;
	}
	else if (dist_metric == geomean)
	{
		distFunc = dist_geomean;
		float eps2 = 1;
		for (int i = 0; i < (*eb)[0].size(); i++)
		{
			eps2 *= eps;
		}
		eps = eps2;
	}
	else
		exit(1);

	try{
		for (unsigned int i = 0; i < eb->size(); i++)
		{
			if (i % 10 == 0)
				cout << i << " ";
			for (unsigned int j = i + 1; j < eb->size(); j++)
			{
				float d = distFunc((*eb)[i], (*eb)[j]);
				if (closethan(d, eps, dist_metric))
				{
					// add edge
					(*this)[i].push_back(j);
					(*this)[j].push_back(i);
					this->nSize += 2;
				}
			}
		}
	}
	catch (...)
	{

	}
	printf("built %d edges\n", this->nSize);
}


// cluster 

Labels Clustering::OneStepCluster(Embeddings* eb, float eps)
{
	printf("OneStepCluster ENTRY\n");

	Edges edges(eb, eps, geomean);

	// init labels
	size_t nNode = eb->size();
	Labels labels(nNode);

	printf("running clustering...\n");

	vector<bool> assigned;
	assigned.assign(nNode, false);

	for (unsigned int i = 0; i < nNode; i++)
	{
		if (!assigned[i])
		{
			assigned[i] = true;
			labels[i] = i;

			for (auto target : edges[i])
			{
				if (assigned[target] == false)
				{
					assigned[target] = true;
					labels[target] = i;
				}
			}
		}
	}
	return labels;
}

Labels Clustering::KMeans(Embeddings* eb, float eps, int k)
{
	printf("KMeans ENTRY\n");

	size_t dim = (*eb)[0].size();
	// init labels

	Centroids centroids(k, dim);
	printf("running clustering...\n");

	//TODO Init centorids
	for (int i = 0; i < k; i++){
		centroids[i] = (*eb)[i];
	}

	return KMeans(eb, centroids, eps, k);
}

Labels Clustering::KMeans(Embeddings* eb, Centroids centroids, float eps, int k)
{
	size_t dim = (*eb)[0].size();
	size_t nNode = eb->size();
	Labels labels(nNode);

	printf("Iterating ");

	int nThread = std::thread::hardware_concurrency();
	ThreadPool pool(nThread);

	bool retry = true;
	int cnt = 0;
	while (retry)
	{
		cnt++;
		printf(".");
		fflush(stdout);

		retry = false;
		// assign each points to nearest label

		vector<future<int>> index_min_f(nNode);
		vector<int> index_min_v(nNode);
		for (int i = 0; i < nNode; i++)
		{
			index_min_f[i] = pool.enqueue(find_min, cref((*eb)[i]), cref(centroids));
		}

		for (int i = 0; i < nNode; i++)
		{
			int index_min = index_min_f[i].get();

			if (index_min != labels[i])
				retry = true;
			labels[i] = index_min;
		}

		// re-evaluate the centers;
		Centroids new_centroids(k, dim);
		for (int i = 0; i < eb->size(); i++)
		{
			new_centroids[labels[i]] += (*eb)[i];
		}

		for (Centroid& centroid : new_centroids)
		{
			centroid.comlete_add();
		}

		centroids = new_centroids;
	}
	cout << endl;


	for (int i = 0; i < eb->size(); i++)
	{
		float dist = dist_euclidean((*eb)[i], centroids[labels[i]]);
		if (dist > eps*eps)
		{
			labels[i] = i;
		}
	}
	printf("Done\n");

	return labels;
}

// cluster in chain manner
Labels Clustering::thresholdCluster(Embeddings* eb, float eps)
{
	printf("thresholdCluster ENTRY\n");
	Edges edges(eb, eps);

	// init labels
	size_t nNode = eb->size();
	Labels labels(nNode);

	printf("running clustering...");

	vector<int> rank(nNode);
	for (auto r : rank)
		r = 0;

	bool retry = true;
	while (retry)
	{
		cout << ".";
		retry = false;
		for (unsigned int i = 0; i < nNode; i++)
		{
			// spread my label
			list<int>::iterator itr;
			for (auto target : edges[i])
			{
				if (labels[i] != labels[target])
				{
					if (rank[labels[i]] > rank[labels[target]])
					{
						labels[target] = labels[i];
					}
					else if (rank[labels[i]] == rank[labels[target]])
					{
						labels[target] = labels[i];
						rank[labels[i]] = rank[labels[i]] + 1;
					}
					else
						labels[i] = labels[target];

					retry = true;
				}
			}
		}
	}

	return labels;
}

int eval_dist(int from, int to, float** dist, Embeddings* eb)
{
	for (int i = from; i < to; i++)
	{
		for (int j = 0; j < i; j++)
		{
			dist[i][j] = dist_euclidean((*eb)[i], (*eb)[j]);
			dist[j][i] = dist[i][j];
		}
	}
	return 0;
}

float** Clustering::init_dist(Embeddings* eb)
{
	cout << "init_dist ENTRY" << endl;

	float** dist;
	size_t nNode = eb->size();
	dist = new float*[nNode];
	for (size_t i = 0; i < nNode; i++)
	{
		dist[i] = new float[nNode];
	}

	size_t nThread = std::thread::hardware_concurrency();
	size_t range = nNode / nThread;
	cout << "evaluating distance... nThread=" << nThread << endl << flush;

	vector<future<int>> flist;
	for (size_t i = 0; i < nThread; i++)
	{
		size_t from = i * range;
		size_t to = (i + 1) * range;
		flist.push_back(async(launch::async, eval_dist, from, to, dist, eb));
	}

	size_t from = nThread * range;
	size_t to = nNode;
	flist.push_back(async(launch::async, eval_dist, from, to, dist, eb));

	for (auto &f : flist)
	{
		f.get();
	}

	for (size_t i = 0; i < nNode; i++)
		dist[i][i] = 0;

	cout << " Done" << endl;
	return dist;
}


vector<Labels> Clustering::Hierarchial(Embeddings* eb, vector<float> epss)
{
	// init labels
	size_t nNode = eb->size();
	vector<Labels> labelVector;
	auto engine = std::default_random_engine{};

	// eval all pair distance
	float** dist = init_dist(eb);

	for (float eps : epss)
	{
		cout << " Eps : " << eps << endl;
		Labels labels(nNode);
		vector<int> nodes;
		for (int i = 0; i < nNode; i++)
			nodes.push_back(i);

		std::shuffle(std::begin(nodes), std::end(nodes), engine);

		list<int>  remains(nodes.begin(), nodes.end());

		// while node left
		while (remains.size() > 0)
		{
			// pick one
			// assign nearby to 
			int core = remains.front();

			for (int other : remains){
				float d = dist[other][core];
				if (d < eps + 0.1)
				{
					labels[other] = core;
				}
			}
			auto isInEps = [dist, core, eps](int target){ return dist[core][target] < eps + 0.1;  };
			remains.erase(remove_if(remains.begin(), remains.end(), isInEps), remains.end());
		}

		labelVector.push_back(labels);
	}

	// return something
	return labelVector;
}

// voca_id -> cluster_id
map<int, int> loadCluster(string path)
{
	map<int, int> dict;
	ifstream infile(path);
	check_file(infile, path);

	string line;
	while (std::getline(infile, line))
	{
		istringstream iss(trim(line));
		int cluster;
		iss >> cluster;
		int item;
		while (!iss.eof()){
			iss >> item;
			dict[item] = cluster;
		}
	}
	return dict;
}


void display(string path, Labels& label, Embeddings* eb)
{
	set<int> distinctLabel(label.begin(), label.end());
	map<int, vector<int>> group;

	for (unsigned int i = 0; i < eb->size(); i++)
	{
		int l = label[i];
		group[l].push_back(i);
	}

	int nNonSingle = 0;


	ofstream fout(path);
	for (auto & v : group)
	{
		if (v.second.size() > 1)
		{
			nNonSingle++;
			fout << v.first << " ";
			for (auto item : v.second)
			{
				fout << (*eb)[item].text << " ";
			}
			fout << endl;
		}
	}

	printf("Number of label : %d\n", distinctLabel.size());
	printf("Number of non single label : %d\n", nNonSingle);

}

void output(Labels& label, size_t nNode, string path)
{
	set<int> distinctLabel(label.begin(), label.end());
	map<int, vector<int>> group;

	for (unsigned int i = 0; i < nNode; i++)
	{
		int l = label[i];
		group[l].push_back(i);
	}

	ofstream fout(path);
	for (auto & v : group)
	{
		if (v.second.size() > 1)
		{
			fout << v.first;
			for (auto item : v.second)
			{
				fout << "," << item;
			}
			fout << endl;
		}
	}

	fout.close();
}

void save_cluster(string path, Embeddings& eb, Word2Idx& word2idx, Labels& labels)
{
	map<int, vector<int>> group;

	for (unsigned int i = 0; i < eb.size(); i++)
	{
		int cluster_id = labels[i];
		int word_id = word2idx[eb[i].text];
		group[cluster_id].push_back(word_id);
	}

	ofstream fout(path);
	for (auto & v : group)
	{
		if (v.second.size() > 1)
		{
			fout << v.first;
			for (auto item : v.second)
			{
				fout << " " << item;
			}
			fout << endl;
		}
	}
	fout.close();
}

/*
parameter :

50 : 10 -> sucks
300 : 30 -> two broad??
*/

void cluster_kmeans()
{
	printf("cluster_kmeans ENTRY\n");
	string path = common_input + "korean_word2vec_wv_300_euckr.txt";
	Embeddings* eb = loadEmbeddings(path.c_str());

	int k = 20;
	float eps = 300;
	string path_param = "parameter.txt";
	ifstream fin(path_param);
	check_file(fin, path_param);
	fin >> k >> eps;
	cout << "k=" << k << " Eps=" << eps << endl;

	map<string, int> word2idx = reverse_idx2word(load_idx2word(common_input + "idx2word"));
	Labels label = Clustering::KMeans(eb, eps, k);
	output(label, eb->size(), data_path + "output.txt");

	// Convert embedding index to voca index
	save_cluster(data_path + "cluster.txt", *eb, word2idx, label);

	display(data_path + "cluster_t.txt", label, eb);

	delete eb;
}

void cluster_embedding()
{
	printf("cluster_embedding ENTRY\n");
	string path = common_input + "korean_word2vec_wv_300_euckr.txt";
	Embeddings* eb = loadEmbeddings(path.c_str());

	//vector<float> epss = { 10, 50, 100, 200, 500, 1000, 2000 };
	vector<float> epss = { 2000, 1000, 500, 100, 10 };

	map<string, int> word2idx = reverse_idx2word(load_idx2word(common_input + "idx2word"));
	vector<Labels> labels = Clustering::Hierarchial(eb, epss);

	for (int i = 0; i < labels.size(); i++)
	{
		//string path_i = data_path + "output_" + std::to_string(i) + ".txt";
		//output(labels[i], eb->size(), path_i);
		string path_c = data_path + "cluster_" + std::to_string(i) + ".txt";
		save_cluster(path_c, *eb, word2idx, labels[i]);
		string path_d = data_path + "cluster_t_" + std::to_string(i) + ".txt";
		display(path_d, labels[i], eb);
	}
}
