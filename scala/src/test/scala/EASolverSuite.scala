import java.io.File

import EntityAssign._
import com.github.tototoshi.csv.CSVReader
import org.scalatest.FunSuite

/**
  * Created by user on 2017-03-01.
  */
class EASolverSuite extends FunSuite {

  test("Dict Test") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val eval = new EAEval("..\\input\\entity_test", dict)

    def known(entity: String): Boolean = {
      dict.has(entity)
    }

    val entitys: List[String] = eval.testCases flatMap (_.entity)

    (entitys filterNot known) foreach println
  }

  test("Extract Generate") {
    val path = "C:\\work\\Data\\reddit\\reddit_cars.csv"
    val reader = new RedditReader(path)
    val dict = new EntityDict("data\\keywords.txt")
    val contents : Stream[String] = reader.data map (reader.content(_))
    val matched : Stream[List[String]] = contents map (dict.extractFrom(_))
    val display = contents zip matched
    matched.slice(0,1000) foreach println
  }

  test("case Test") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val eval = new EAEval("..\\input\\entity_test", dict)

    val entitys: List[String] = eval.testCases flatMap (_.entity)

    //entitys foreach println
    assert(entitys forall (_.length > 0))
  }

  test("Compare Accuracy") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val path = "..\\input\\CarAffinity.txt"
    val eval = new EAEval("..\\input\\entity_test1", dict)

    val solverTarget = new TargetOnly(dict)
    val accuracyTarget = eval.evalPerformance(solverTarget)
    println(s"Accuracy[Target] : $accuracyTarget")

    val solver1 = new Recent(dict)
    val accuracyRecent = eval.evalPerformance(solver1)
    println(s"Accuracy[Recent] : $accuracyRecent")

    val solver2 = new RecentsFirst(dict)
    val accuracyRecentFirst = eval.evalPerformance(solver2)
    println(s"Accuracy[Recent(One Entity)] : $accuracyRecentFirst")

    val solver3 = new FirstOnly(dict)
    val accuracyFirst = eval.evalPerformance(solver3)
    println(s"Accuracy[First Only]   : $accuracyFirst")

    val solverContext = new EntityContext(dict)
    val accuracyContext = eval.evalPerformance(solverContext)
    println(s"Accuracy[Context]   : $accuracyContext")
//
//    val solverContextC = new EAContextCascade(dict)
//    val accuracyContextC = eval.evalPerformance(solverContextC)
//    println(s"Accuracy[ContextC]   : $accuracyContextC")

    val mesolver = new MESolver(dict, eval.testCases)
    val accuracyStruct = eval.evalPerformance(mesolver)
    println(s"Accuracy[ME]   : $accuracyStruct")

  }

  test("Compare precision/recall"){
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val eval = new EAEval("..\\input\\entity_test1", dict)

    val solverTarget = new TargetOnly(dict)
    val solverRecent = new Recent(dict)
    val solverFirst = new RecentsFirst(dict)
    val solverFirstOnly = new FirstOnly(dict)
    val mesolver = new MESolver(dict, eval.testCases)

    val list : List[(EASolver, String)] = List((solverTarget,"Target"), (solverRecent, "Recent"), (solverFirst,"First"),
        (solverFirstOnly,"FirstOnly"), (mesolver,"ME"))

    list foreach { x =>
      val name = x._2
      val (precision, recall) = eval.getRecallPrecision(x._1)
      val f = 2 * precision * recall / ( precision + recall + 0.01)
      println(s"Accuracy[$name] : $precision , $recall , $f")
    }
  }


  test("Affinity - Threshold") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val path = "..\\input\\CarAffinity.txt"
    val affinityDict = new Affinity(path, dict)

    val eval = new EAEval("..\\input\\entity_test", dict)
    val solver = new AffinityThreshold(dict, affinityDict)
    val accuracy = eval.evalPerformance(solver)

    println(s"Accuracy : $accuracy")
  }

  test("ContextModel") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val eval = new EAEval("..\\input\\entity_test", dict)
    val solver = new EAContextCascade(dict)
    eval.showResult(solver)
    val accuracy = eval.evalPerformance(solver)
    println(s"Accuracy : $accuracy")
  }

  test("Show Context ") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val path = "..\\input\\CarAffinity.txt"

    val eval = new EAEval("..\\input\\entity_test", dict)
    val solver = new EntityContext(dict)
    eval.showResult(solver)
  }
}