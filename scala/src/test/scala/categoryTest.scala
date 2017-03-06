

import org.scalatest.FunSuite
import sfc._
import sfc.category._
import sfc.tag._
import sfc.list._
import sfc.sfc2._
import stringHelper._

import scala.io.Source

class CategorySuite extends FunSuite {

  test("Tokenize Test"){
    val sentence1 = "결국에 현기 사는 이유는 뭘 제대로 알아 보고 산게 아니라 많이 팔리니까 사는거라는 소린가요?? "
    val sentence2 = "저는 예쁜데 왜 그러지 ㅠ"
    val res1 = tokenize(sentence2)
    res1 foreach (x=> print(x+"."))
  }

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
    val scfDict : SCFDictionary = scfDictBest(tags)
    def checker(text : String) : Boolean = {
      val result = isComplete(categorys)(scfDict)(text)
      if(!result)
      {
        println(s"$text is not complete sentence")
        val remain : Iterable[Argument] = allUnmatchedArg(categorys)(scfDict)(text)
        println("Remaining arguments are :")
        remain foreach {
          _.print
        }
      }

      result
    }
    assert(checker("아버지는 오토바이를 탔다") === true)
    assert(checker("집이 탔다") === false)
    assert(checker("") === true)
    assert(checker("님이 한번 중형차 타보시면 압니다.") === true)
    assert(checker("저는 예쁜데 왜 그러지 ㅠ")=== false)


    def testSFC(sentence: String) = {
      val matches =  applyPossibleSCF(categorys)(scfDict)(sentence)
      matches foreach (_.print)
    }

    testSFC("저는 예쁜데 왜 그러지 ㅠ")
    testSFC("군인은 꽃을 예쁘다고 생각합니다.")

    val ceylon = new Ceylon(categorys, scfDict)

    ceylon.showRecovery("직원은 예쁘다고 하죠.", "소나타랑 트럭중에 어느게 예뻐요?")
  }

  test("Recovery on Bobae"){

    val categorys : List[(String, Category)] = loadCategory("data\\StdDic.txt", "data\\Synset.txt", "data\\Tran.txt")
    val tags = allTags(categorys map (_._2))
    val scfDict : SCFDictionary = scfDictBest(tags)
    val ceylon = new Ceylon(categorys, scfDict)

    val rootInput = "C:\\work\\Code\\Community-Sentiment-Analysis\\input\\"
    val filename = "related.txt"
    val path: String =rootInput + filename
    for (line <- Source.fromFile(path).getLines()) {
      val cutIdx = line.indexOf('-')
      if(cutIdx > 0){
        val target = line.substring(0,cutIdx)
        val context = line.substring(cutIdx + 1)
        println("---------------")
        ceylon.showRecovery(target, context)
      }
    }
  }
}
