/**
  * Created by user on 2016-12-28.
  */
package sfc

import java.io.File

import sfc.tag._
import com.github.tototoshi.csv._

package object category{

  type Category = List[Tag]
  type TaggedTokens = Seq[(String, Option[Category])]

  def printCategory(c : Category) = {
    c foreach ( x => print(x.name + " or ") )
  }

  def loadCSV(path :String) : Stream[List[String]] = {
    val reader = CSVReader.open(new File(path), "euc-kr")
    val stream : Stream[List[String]] = reader.toStream
    stream
  }

  def allTags(categorys : Iterable[Category]) : Set[Tag] = {
    categorys.flatten.toSet
  }

  // StdDic : Standard Dictionary : Human words
  // Synset : Symentic Set : Categorical information
  // tran : Translation table (Std-Syn matching)
  def loadCategory(pathStdDic: String, pathSynset:String, pathTran:String) : List[(String, Category)] = {
    def parseStdDic( l : List[String]) : (Int, String) = {
      (l.head.toInt, l.tail.head)
    }
    val stdDic : Stream[(StdIdx, String)] = loadCSV(pathStdDic) map parseStdDic
    val stdDicMap : Map[StdIdx,String] = stdDic.toList.toMap


    def parseSynset( l : List[String]) : (StdIdx, SynsetOffset, String, List[SynsetOffset]) = {
      def parseHypernym(str : String) : SynsetOffset = str.split("_")(0).toInt

      val arr = l.toArray
      val synsetidx = arr(0).toInt
      val offset = arr(1).toInt
      val word: String = arr(2)
      val hypernymStrs : List[SynsetOffset] = {
        if(arr(4).isEmpty)
          Nil
        else
          (arr(4).split(",") map parseHypernym).toList
      }

      (synsetidx, offset, word, hypernymStrs)
    }
    val synset : Stream[(SynsetIdx, SynsetOffset, String, List[SynsetOffset]) ] = loadCSV(pathSynset) map parseSynset
    val parentsOf : Map[SynsetOffset, List[SynsetOffset]] = (synset.toList map (x => (x._2, x._4) )).toMap
    val synsetIdx2offset : Map[SynsetIdx, SynsetOffset] = (synset.toList map (x => (x._1, x._2) )).toMap
    val nameOf : Map[SynsetOffset, String] = (synset.toList map (x => (x._2, x._3) )).toMap

    def parseTran( l : List[String]) : (Int, SynsetIdx, StdIdx, String) = {
      val arr = l.toArray
      val tranidx = arr(0).toInt
      val synsetidx = arr(1).toInt
      val stdDicidx = arr(3).toInt
      val word : String = arr(6)
      (tranidx, synsetidx, stdDicidx, word)
    }
    val tran : Stream[(Int, SynsetIdx, StdIdx, String) ] = loadCSV(pathTran) map parseTran
    val synsetOffset2Name : Map[SynsetOffset, String] = (tran map (x => (synsetIdx2offset(x._2), x._4))).toMap
    val stdIdx2SynsetOffset : Map[StdIdx, SynsetOffset] = (tran map (x => (x._3, synsetIdx2offset(x._2)))).toMap
    // for every string in stdDic
    // find matching stddicIdx in trans
    // travel along wordnetidx to extract all word set
    // add all tag set to
    def getNameStr(synsetOffset: SynsetOffset) : String = {
      val synsetStr = nameOf(synsetOffset)
      val transName = {
        if(synsetOffset2Name.contains(synsetOffset))
          synsetOffset2Name(synsetOffset)
        else
          "-"
      }
      synsetStr + " " + transName
    }
    def allTagsFor(stdDicEntry : (StdIdx, String)) : Category = {
      val stdidx = stdDicEntry._1
      val synsetOffset : SynsetOffset = stdIdx2SynsetOffset(stdidx)

      def follow(first : List[SynsetOffset]) : List[SynsetOffset] = {
        if (first.isEmpty)
          Nil
        else {
          val parents: List[SynsetOffset] = (first map parentsOf).flatten
          first ++ follow(parents)
        }
      }

      val tagsIDs : List[SynsetOffset] = follow(List(synsetOffset))
      tagsIDs map (x => new Tag(x, parentsOf(x), getNameStr(x) ) )
    }

    val validStdDIc = stdDic filter (x => stdIdx2SynsetOffset.contains(x._1))
    val strm = validStdDIc map ( x=> (x._2, allTagsFor(x)))
    strm.toList
  }

  def tagger(tagList : List[(String, Category)])(word: String) : Option[Category]= {
    lazy val tagMap = tagList.toMap
    if( tagMap.contains(word) )
      Some(tagMap(word))
    else
      None
  }

  def printTokenizeResult(taggedTokens: TaggedTokens) = {
    def printResult(pair: (String, Option[Category])) = {
      print(pair._1 + ":")
      pair._2 match {
        case Some(category) => printCategory(category)
        case None => print("None")
      }
      println("")
    }
    taggedTokens foreach printResult
  }
}