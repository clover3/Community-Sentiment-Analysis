
import java.nio.charset.StandardCharsets

import org.scalatest.FunSuite
import category._
import com.twitter.penguin.korean.TwitterKoreanProcessor
import scf._
import stringHelper._


class CategorySuite extends FunSuite {

  test("check category load"){
    val categories = loadCategory("data\\StdDic.txt", "data\\Synset.txt", "data\\Tran.txt")

    def printer(entry : (String, Category)) = {
      print(entry._1)
      print(":")
      printCategory(entry._2)
      println("")
    }
    val mother = categories filter ( x=> x._1 contains "차")
  }

  def loadIdx2word(path : String) : Map[Int, String] = {
    val fileLines : Iterator[String] = io.Source.fromFile(path).getLines


    def parseLine(line:String) : (Int, String)= {
      val tokens = line.split("\t")
      val idx = tokens(0).toInt
      val word = tokens(1)
      (idx, word)
    }
    val idx2word: Map[Int, String] = (fileLines map parseLine).toMap
    idx2word
  }

  test("check how many bobae tokens matches any category ")
  {
    val path = "C:\\work\\Code\\Community-Sentiment-Analysis\\input\\idx2word_utf.txt"
    val idx2word : Map[Int, String] = loadIdx2word(path)

    val words : Iterable[String] = idx2word.values
    val categoryMap : Map[String, Category] = loadCategory("data\\StdDic.txt", "data\\Synset.txt", "data\\Tran.txt").toMap


    val matchedWords = words.filter(categoryMap.contains)
    val unmatchedWords = words.filterNot(categoryMap.contains)
    val matched = matchedWords.size
    val total = words.size
    println(s"Total words = $total matched = $matched")
  }

  test("Tagger Check")
  {
    val categorys : List[(String, Category)] = loadCategory("data\\StdDic.txt", "data\\Synset.txt", "data\\Tran.txt")
    def myTagger = tagger(categorys)(_)
    val sentence1 = "아버지는 오토바이를 탔다."
    def tag(text: String) : TaggedTokens = {
      val tokens: Seq[String] = tokenize(text)
      val result: Seq[Option[Category]] = tokens map myTagger
      tokens zip result
    }
    printTokenizeResult(tag(sentence1))
    printTokenizeResult(tag("선생님은 포드를 탔다"))
    printTokenizeResult(tag("집이 탔다"))
  }


  test("SCF function check")
  {
    val categorys : List[(String, Category)] = loadCategory("data\\StdDic.txt", "data\\Synset.txt", "data\\Tran.txt")
    val tags = allTags(categorys map (_._2))
    val tagFinder = new TagFinder(tags)

    val personTag : Tag = tagFinder.findByName(TagHumanLike)
    val actor : Argument = new Argument(personTag, "Actor")

    val vehicleTag : Tag = tagFinder.findByName(TagVehicle)
    val vehicle : Argument = new Argument(vehicleTag, "Target")
    val ride : SCF = new SCF("타다", List(actor, vehicle))

    val scfDict : SCFDictionary = new SCFDictionary(List(ride))
    def checker :(String)=>Boolean = isComplete(categorys)(scfDict)(_)
    assert(checker("아버지는 오토바이를 탔다") === true)
    assert(checker("집이 탔다") === false)
    assert(checker("") === true)
    assert(checker("님이 한번 중형차 타보시면 압니다.") === true)
  }

}
