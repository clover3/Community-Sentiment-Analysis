import java.io.{BufferedReader, FileInputStream, InputStreamReader}
import java.nio.charset.CodingErrorAction
import java.util.Dictionary

import scala.io.Codec
import scala.reflect.io.File


object Indexrize {

  val decoder = Codec.UTF8.decoder.onMalformedInput(CodingErrorAction.IGNORE)
  def readLines(path: String) : Stream[String] = {
    val in_file = new FileInputStream(path)
    val in_stream = new InputStreamReader(in_file, "MS949")
    val in = new BufferedReader(in_stream)
    val strs = Stream.continually(in.readLine()).takeWhile(_ != null)
    strs map (_.trim) filter(_.length() > 0)
  }

  def saveLines(path: String, lines: Iterable[String]) : Unit = {
    var out_file = new java.io.FileOutputStream(path)
    var out_stream = new java.io.PrintStream(out_file)
    lines foreach out_stream.println
    out_stream.close()
  }
  def loadIdx2Word(path:String): Map[Int, String] = {
    val lines = readLines(path)
    def parseLine(rawString : String) : (Int, String) = {
      val tokens : Array[String] = rawString.trim().split("\\s+")
      val idx : Int = tokens(0).toInt
      val word : String = tokens(1)
      (idx,word)
    }
    val tLine : Stream[(Int, String)] = lines map parseLine
    val parsedLine : Map[Int, String] = tLine.toMap
    parsedLine
  }

  def convert2readable(pathTokenedFile:String, idx2word : Map[Int, String] , outPath :String) : Unit = {
    val lines = readLines(pathTokenedFile)
    def parseLine(rawString : String) : Array[Int] = {
      val tokens : Array[String] = rawString.trim().split("\\s+")
      tokens map (x=>x.toInt)
    }
    val wordList : Stream[Array[String]] = lines map (x => parseLine(x) map idx2word)
    val strings : Stream[String] = wordList map (x => x.mkString(" "))
    saveLines(outPath, strings)
  }

  def main(args: Array[String]): Unit = {
    println("Hi Scala")
    val idx2word = loadIdx2Word("..\\input\\idx2word")
    convert2readable("..\\input\\recovered.index", idx2word, "recovered.text")

  }
}
