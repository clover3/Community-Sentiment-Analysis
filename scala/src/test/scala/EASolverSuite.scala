import EntityAssign._
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

  test("case Test") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")
    val eval = new EAEval("..\\input\\entity_test", dict)

    val entitys: List[String] = eval.testCases flatMap (_.entity)

    //entitys foreach println
    assert(entitys forall (_.length > 0))
  }

  test("Compare Accuracy") {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val eval = new EAEval("..\\input\\entity_test", dict)

    val solverTarget = new TargetOnly(dict)
    val accuracyTarget = eval.evalPerformance(solverTarget)
    println(s"Accuracy[Target] : $accuracyTarget")

    val solver1 = new Recent(dict)
    val accuracyRecent = eval.evalPerformance(solver1)
    println(s"Accuracy[Baseline1] : $accuracyRecent")

    val solver2 = new RecentsFirst(dict)
    val accuracyRecentFirst = eval.evalPerformance(solver2)
    println(s"Accuracy[Baseline2] : $accuracyRecentFirst")

    val solver3 = new FirstOnly(dict)
    val accuracyFirst = eval.evalPerformance(solver3)
    println(s"Accuracy[First Only]   : $accuracyFirst")

    val structSolver = new StructureSolver(dict, eval.testCases)
    val accuracyStruct = eval.evalPerformance(structSolver)
    println(s"Accuracy[Struct]   : $accuracyStruct")

    val solverContext = new EntityContext(dict)
    val accuracyContext = eval.evalPerformance(solverContext)
    println(s"Accuracy[Context]   : $accuracyContext")

    //eval.showResult(solver = solver1)
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
    val path = "..\\input\\CarAffinity.txt"

    val eval = new EAEval("..\\input\\entity_test", dict)
    val solver = new EntityContext(dict)
    val accuracy = eval.evalPerformance(solver)

    println(s"Accuracy : $accuracy")
  }
}