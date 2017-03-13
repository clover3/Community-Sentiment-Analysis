import EntityAssign.{ContextModel, EACase, EAEval, EntityDict}
import org.scalatest.{FunSuite, fixture}

/**
  * Created by user on 2017-03-11.
  */
class ContextModelSuite extends FunSuite {
  test("Model Init")
  {
    val dict = new EntityDict("C:\\work\\Code\\Community-Sentiment-Analysis\\input\\EntityDict.txt")

    val eval = new EAEval("..\\input\\entity_test", dict)
    val contextModel = new ContextModel(dict)
    val testCases:Array[EACase] = eval.testCases.toArray
    val targetCase = (testCases find (x => x.targetSent contains("재질은 말랑말랑해요"))).get
    println(dict.getName(141))
    println(targetCase.targetSent)
    contextModel.contextWords(targetCase,141) foreach println
  }
}
