import EntityAssign.{EAEval, EntityDict, FirstOnly, RecentFirst}
import org.scalatest.FunSuite

/**
  * Created by user on 2017-03-01.
  */
class EASolverSuite extends FunSuite{

  test("RecentFirst"){
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val eval = new EAEval("..\\input\\entity_test", dict)

    val solver1 = new RecentFirst(dict)
    val accuracyRecent = eval.evalPerformance(solver1)

    val solver2 = new FirstOnly(dict)
    val accuracyFirst = eval.evalPerformance(solver2)

    println(s"Accuracy[Recent First] : $accuracyRecent")
    println(s"Accuracy[First Only]   : $accuracyFirst")

    eval.showResult(solver1)
  }
}
