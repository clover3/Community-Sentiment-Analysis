import EntityAssign.EntityDict
import org.scalatest.FunSuite

/**
  * Created by user on 2017-03-01.
  */
class EntityDictSuite extends FunSuite {
  test("EntityDict Test") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val groupAvante = dict.entity2group("아방이")
    val groupAvante2 = dict.entity2group("아반떼")
    assert(groupAvante === groupAvante2)

    dict.group(groupAvante) foreach println
  }

}
