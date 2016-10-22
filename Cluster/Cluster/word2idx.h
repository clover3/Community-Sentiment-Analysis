#pragma once
#include "stdafx.h"


using Idx2Word = map <int, string>;
using Word2Idx = map <string, int>;

Idx2Word load_idx2word(string path);
map<string, int > reverse_idx2word(Idx2Word& idx2word);