import EntityAssign.{Affinity, EntityDict}
import org.scalatest.FunSuite

/**
  * Created by user on 2017-03-11.
  */
class AffinitySuite extends FunSuite {
  test("Test - 1")
  {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val path = "..\\input\\CarAffinity.txt"
    val affinityDict = new Affinity(path, dict)

    val car = "소나타"
    val word = "이런단어는존재할수가없어"
    val f = affinityDict.get(car, word)
    assert(f == 1 )
    println(s"affinity($car, $word) == $f")

    val car2 = "SM7"
    val word2 = "르노"
    val f2 = affinityDict.get(car2, word2)
    assert(affinityDict.get(car2, word2) > 4 )
    println(s"affinity($car2, $word2) == $f2")

    assert(affinityDict.get("현기", "호갱") > 1.5 )
  }

}
