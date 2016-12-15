#pragma once
#include "stdafx.h"

class Embedding : public vector < float >
{
public:
	string text;
};
using Embeddings = vector<Embedding>;

Embeddings* loadEmbeddings(const char* path);
float dist_euclidean(const vector<float> &e1, const  vector<float> &e2);
float dist_manhattan(const Embedding &e1, const Embedding &e2);
float dist_geomean(const Embedding &e1, const Embedding &e2);