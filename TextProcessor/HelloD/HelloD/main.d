import std.container: SList;
import std.algorithm;
import std.array;
import std.csv;
import std.stdio;
import std.typecons;
import std.string;
import std.conv;
import std.file;

class IDTable{
	int[string] dic;
	int[string] count;
	string[] revdic;
	int idx = 0;
	
	void addIfNotIn(string key)
	{
		int* p ;
		p = (key in dic);
		if(p is null)
		{
			dic[key] = idx;
			revdic ~= key;
			idx += 1;
			count[key] = 0;
		}
		count[key] += 1;
	}

	int opIndex(string key)
	{
		return dic[key];
	}
	
	int get_index(string key)
	{
		addIfNotIn(key);
		return dic[key];
	}
}



void readcsv(string path)
{
	string content = readText(path);
	auto file = File(path, "r");
	auto records = csvReader!(Tuple!(string, string, string, string, string, 
									 string, string, string, string, string, string), Malformed.ignore)(content);
	auto fout = File("output.txt", "w");
	IDTable tokenDic = new IDTable;
	foreach(record;  records)
	{
		auto tokens = split(record[10],"/");
		foreach(token; tokens)
		{
			tokenDic.addIfNotIn(token);
			fout.writef("%s ", token);
		}
		fout.writef("\n");
	}
		
}

// Complete Sentence Table = 


class Document{
// Pair of Array[Article] and Dictionary of term_id to 
	Article[] articles;
	IDTable index;

	this()
	{
		index = new IDTable;
	}
	void addArticle(Article a)
	{
		articles ~= a;
	}

	void updateTokenDic(string token)
	{
		index.addIfNotIn(token);
	}

}


// Sentence : = Array[int]



alias Sentence = int[];

bool less(Sentence s1, Sentence s2)
{
	if( s1.length != s2.length)
		return s1.length < s2.length;

	foreach(int i , int e; s1)
	{
		if( s1[i] != s2[i] )
			return s1[i] < s2[i];
	}
	return false;
}

bool equal(Sentence s1, Sentence s2)
{
	if( s1.length != s2.length)
		return false;

	sort(s1);
	sort(s2);

	foreach(int i , int e; s1)
	{
		if( s1[i] != s2[i] )
			return false;
	}
	return true;
}

void printSentence(Sentence s)
{
	writef("len=%d : ", s.length);
	foreach(token; s)
	{
		writef("%d ",token);
	}
	write("\n");
}

void printSentenceWord(Sentence s, IDTable dic)
{
	foreach(token; s)
	{
		writef("%s ",dic.revdic[token]);
	}
	write("\n");
}

class Article {
	Sentence[] sentences;
	Article[] childs;
	int type;
	int article_id;
	int thread_id;
	string content;

	Sentence getWholeSentence()
	{
		Sentence total;
		foreach(s; sentences)
		{
			total ~= s.dup();
		}
		return total;
	}

	this(Sentence[] ss, string content, int type, int article_id, int thread_id)
	{
		this.sentences = ss;
		this.content = content;
		this.type = type;
		this.article_id = article_id;
		this.thread_id = thread_id;
	}
	void print()
	{
		writefln("Article type=%d", type);
		foreach(s ; sentences)
		{
			printSentence(s);
		}
	}
}


Sentence init_sentence(string[] raw_tokens, IDTable dic)
{
	Sentence s;
	s.length= raw_tokens.length;
	int index = 0;
	foreach(raw_token ; raw_tokens)
	{
		int id = dic[raw_token];
		s[index] = id;
		index ++;
	}
	return s;

}

// CS Table 을 요약하자.

class CSTable
{
// CSTable 멤버 변수
	class CSEntry
	{
		int id;
		Sentence s;
		int count;
		this(Sentence cs, int index){
			this.id = index;
			this.s = cs;
			this.count = 1;
		}
	}

	CSEntry[] list;
	Document doc;
// CSTable 멤버 함수
	// csArr
	this(Document doc, Sentence[] csArr)
	{
		this.doc = doc;
		CSEntry recent;

		int index = 0;
		sort!(less)(csArr);
		foreach(cs; csArr)
		{
			if( recent !is null && equal(cs, recent.s) ) // 이전과 같으면 카운트만 올려준다.
			{
				recent.count++;
			}
			else
			{
				CSEntry entry = new CSEntry(cs, index++);
				list ~= entry;
				recent = entry;
			}
		}
	}

	void printFile()
	{
		File f = File("cstable.txt", "w");
		foreach(entry; list)
		{
			f.writef("[%d]-%d : ", entry.id, entry.count);
			foreach(token; entry.s)
			{
				f.write(doc.index.revdic[token] , " ");
			}


			f.write("\n");
		}
	}
}

// CSV 를 읽어서 article 의 배열을 반환한다.
Document csvToArticleArray(string path)
{
	Document doc = new Document;
	
	string content = readText(path);
	auto records = csvReader!(Tuple!(string, string, string, string, 
									string, string, string, string, 
									string, string, string), Malformed.ignore)(content);
	
	auto fout = File("output.txt", "w");
	foreach(record; records)
	{
		Sentence[] articleSentence;
		// |를 문장의 Delimeter 로 하여 끊는다.
		auto delimited_sentences = split(record[10],"|");
		// 문장의 token 을 계산한다.
		foreach(raw_sentence; delimited_sentences)
		{
			string[] tokens = split(raw_sentence,"/");	
			foreach(token; tokens)
			{
				doc.index.addIfNotIn(token);
			}
			// Sentene 객체를 만들고 넣는다.
			Sentence s = init_sentence(tokens, doc.index);
			articleSentence ~= s;
		}

		string raw_content = record[2];
		int article_id = to!uint(record[5]);
		int thread_id = to!uint(record[6]);
		int type = to!uint(record[9]);
		Article article = new Article(articleSentence, raw_content, type, article_id, thread_id );
		doc.articles ~= article;
	}
	return doc;
}	


void Anal(string path)
{
	Document doc = csvToArticleArray(path);

	int cnt = 0;
	
	foreach(article; doc.articles)
	{
	}
}

CSTable create_cstable(string path)
{
	Document doc = csvToArticleArray(path);
	int level = 0;


	Sentence[] cslist;

	foreach(article ; doc.articles)
	{
		if(article.type == 0)
		{
			Sentence cs_elem = article.getWholeSentence().dup();
			cslist ~= cs_elem;
			if( cs_elem.length == 1)
				writefln("%d ", article.article_id);
		}
	}	
	
	CSTable table = new CSTable(doc, cslist);
	return table;
}

// Complete Sentence Table 을 생성한다.
CSTable create_cstable_sentence_level(string path)
{
	Document doc = csvToArticleArray(path);
	int level = 0;


	Sentence[] cslist;
	Article[] stack;
	void update_table()
	{	
		// Update Table
		Sentence cs;
		int cnt = 0;
		foreach(artc ; stack)
		{
			foreach(s ; artc.sentences)
			{
				cs ~= s;
				if( cs.length < 30 )
				{
					// cs 를 복사해서 table 에 집어 넣어야 함.
					Sentence cs_elem = cs.dup;
					cslist ~= cs_elem;	
					cnt++;
				}
			}
		}
	}
	// 각 article 을 Looping하면서 stack 을 쌓는다.
	foreach(article ; doc.articles)
	{
		// Level 에 맞을때까지 Pop 을 한다.
		bool popRequired = article.type < stack.length;

		if( popRequired )
		{
			update_table();

			while(article.type < stack.length)
				stack.length --;
		}
		// Stack 을 쌓는다.
		stack ~= article;
	}
	update_table();


	CSTable table = new CSTable(doc, cslist);
	return table;
}


void extract_first_sentence(string path)
{
	Document doc = csvToArticleArray(path);
	foreach(article; doc.articles)
	{
		if( article.type == 0)
		{
			foreach(s ; article.sentences)
			{
				
			}
		}
	}

}

int main(string[] argv)
{
	create_cstable("babae_car_tokens_utf.csv").printFile();
	return 0;
}	