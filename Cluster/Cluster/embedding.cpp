#include "embedding.h"

Embeddings* loadEmbeddings(const char* path)
{
	Embeddings *ptr = new Embeddings;
	FILE* fp;
#ifdef WINVS
	fopen_s(&fp, path, "r");
#else
	fp = fopen(path, "r");
#endif
	if (!fp)
	{
		cout << "file open failed" << endl;
		exit(0);
	}

	int nEntry, nDim;
	string str;
	fscanf_s(fp, "%d %d", &nEntry, &nDim);

	//	nEntry = 10000;
	printf("Entry :%d \nDimension : %d\n", nEntry, nDim);
	ptr->resize(nEntry);

	cout << "Loading embedding..";
	int cnt = 0;
	char buf[4000];
	for (int i = 0; i < nEntry; i++)
	{
		Embedding& v = (*ptr)[i];
		v.resize(nDim);
		fscanf_s(fp, "%s", buf);
		v.text = buf;

		for (int j = 0; j < nDim; j++)
		{
			float f;
			fscanf_s(fp, "%f ", &f);
			v[j] = f;
		}
	}
	fclose(fp);


	cout << " done " << endl;
	return ptr;
}


float dist_euclidean(const vector<float> &e1, const  vector<float> &e2)
{
	float acc = 0;
	for (unsigned int i = 0; i < e1.size(); i++)
	{
		float d = (e1[i] - e2[i]);
		acc += d * d;
	}
	return acc;
}

float dist_manhattan(const Embedding &e1, const Embedding &e2)
{
	float acc = 0;
	for (unsigned int i = 0; i < e1.size(); i++)
	{
		float d = fabs(e1[i] - e2[i]);
		acc += d;
	}
	return acc;
}

float dist_geomean(const Embedding &e1, const Embedding &e2)
{
	float acc = 1;
	for (unsigned int i = 0; i < e1.size(); i++)
	{
		float d = fabs(e1[i] - e2[i]);
		acc *= d;
	}
	return acc;
}