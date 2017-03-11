import EntityAssign._
import org.scalatest.FunSuite

/**
  * Created by user on 2017-03-01.
  */
class EASolverSuite extends FunSuite{

  test("Dict Test"){
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val eval = new EAEval("..\\input\\entity_test", dict)

    def known(entity: String) : Boolean = {
      dict.has(entity)
    }

    val entitys:List[String] = eval.testCases flatMap (_.entity)

    (entitys filterNot known) foreach println
  }

  test("case Test"){
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val eval = new EAEval("..\\input\\entity_test", dict)

    val entitys:List[String] = eval.testCases flatMap (_.entity)

    //entitys foreach println
    assert(entitys forall (_.length > 0 ))
  }

  test("RecentFirst"){
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val eval = new EAEval("..\\input\\entity_test", dict)

    val solver1 = new Recent(dict)
    val accuracyRecent = eval.evalPerformance(solver1)

    val solver2 = new RecentsFirst(dict)
    val accuracyRecentFirst = eval.evalPerformance(solver2)

    val solver3 = new FirstOnly(dict)
    val accuracyFirst = eval.evalPerformance(solver3)

    println(s"Accuracy[Baseline1] : $accuracyRecent")
    println(s"Accuracy[Baseline2] : $accuracyRecentFirst")
    println(s"Accuracy[First Only]   : $accuracyFirst")

    //eval.showResult(solver = solver1)
  }
}
