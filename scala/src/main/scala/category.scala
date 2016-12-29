/**
  * Created by user on 2016-12-28.
  */
import java.io.File

import com.github.tototoshi.csv._

object category{

  trait Tag {
    val parent : Tag
    val name : String
  }

  trait Category {
    val tags : List[Tag]
  }

  def loadCSV(path :String) : Stream[List[String]] = {
    val reader = CSVReader.open(new File(path))
    val stream : Stream[List[String]] = reader.toStream
    stream
  }

  type WordID = Int
  type TagID = Int
  type Hypernym = TagID

  def loadCategory(pathStdDic: String, pathSynset:String, pathTran:String) : Map[String, Category] = {
    def parseStdDic( l : List[String]) : (Int, String) = {
      (l.head.toInt, l.tail.head)
    }
    val stringTokens : Stream[(Int, String)] = loadCSV(pathStdDic) map parseStdDic
    val stdDic : Map[Int,String] = stringTokens.toList.toMap


    def parseSynset( l : List[String]) : (TagID, String, Array[Hypernym]) = {
      def parseHypernym(str : String) : Hypernym = str.split("_")(0).toInt

      val arr = l.toArray
      val offset = arr(1).toInt
      val word : String = arr(2)
      val hypernymStrs : Array[Hypernym] = arr(4).split(",") map parseHypernym

      (offset, word, hypernymStrs)
    }
    val synset : Stream[(TagID, String, Array[Hypernym]) ] = loadCSV(pathStdDic) map parseSynset

    val tran : Stream[(TagID, String, Array[Hypernym]) ] = loadCSV(pathTran) map parseSynset
  }

}