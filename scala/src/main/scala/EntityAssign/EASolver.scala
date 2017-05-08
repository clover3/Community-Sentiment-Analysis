package EntityAssign

import stringHelper.{filterStopWords, keyTokens, tokenize, tokenizeWithPos}
import EntityAssign.EID.EntityID
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken
import com.twitter.penguin.korean.util.KoreanPos.KoreanPos
import maxent.MaxEnt


/**
  * Created by user on 2017-03-11.
  */

trait EASolver {
  def solve(testCase : EACase) : List[String]
}


class EATool(dict: EntityDict) {
  def union(e1 : List[String], e2 : List[String]) : List[String] =
  {
    val sum : Set[Int] = (e1 map dict.getGroup).toSet ++ (e2 map dict.getGroup).toSet
    sum.toList map dict.getName
  }
  def lastMentionedEntitys(contexts: List[String]): List[String] = {
    // If tail has Some give it, if tail None, then do at the head
    contexts match {
      case Nil => Nil
      case head :: tail => {
        val preEntity = lastMentionedEntitys(tail)
        if (preEntity == Nil)
          dict.extractFrom(head)
        else
          preEntity
      }
    }
  }

  def prevSentence(contexts: List[String]): Option[String] = {
    contexts match {
      case Nil => None
      case head :: tail => {
        prevSentence(tail) match {
          case Some(s) => Some(s)
            val headEntity = dict.extractAnyFrom(head)
            if (headEntity.isDefined)
              Some(head)
            else
              None
          case None => None
        }
      }
    }
  }

  def firstEntity(texts : List[String]) : List[String] = {
    texts match {
      case Nil => Nil
      case head :: tail => {
        val headEntity = dict.extractFrom(head)
        if (headEntity.isEmpty)
          firstEntity(tail)
        else
          headEntity
      }
    }
  }

  def isMostFrequent(text: List[String], entity:String) : Boolean = {
    val entityAll:List[Int] = text flatMap dict.extractFrom map dict.getGroup
    if(entityAll.isEmpty)
      false
    else
    {
      val mostFrequent = entityAll.groupBy(identity).maxBy(_._2.size)._1
      mostFrequent == dict.getGroup(entity)
    }
  }

  def allEntity(text : List[String]) : Set[String] = {
    (text flatMap dict.extractFrom).toSet
  }

  def allEntityAsID(text : List[String]) : Set[EntityID] = {
    allEntity(text) map dict.getGroup
  }
}


// Baseline 1
class Recent(entityDict: EntityDict) extends EASolver {
  val tool = new EATool(entityDict)
  override def solve(testCase: EACase): List[String] = {
    val entityOfTarget : List[String] = entityDict.extractFrom(testCase.targetSent)
    if (entityOfTarget != Nil)
      entityOfTarget
    else{
      tool.lastMentionedEntitys(testCase.context)
    }
  }
}

class TargetOnly(entityDict: EntityDict) extends EASolver {

  override def solve(testCase: EACase): List[String] = {
    entityDict.extractFrom(testCase.targetSent)
  }

}

class RecentsFirst(entityDict: EntityDict) extends Recent(entityDict) {
  override def solve(testCase: EACase): List[String] = {
    val entityOfTarget : List[String] = entityDict.extractFrom(testCase.targetSent)

    if (entityOfTarget != Nil)
      entityOfTarget
    else{
      tool.lastMentionedEntitys(testCase.context) match {
        case head::tail => List(head)
        case Nil => Nil
      }
    }
  }
}

class FirstOnly(entityDict: EntityDict) extends EASolver {
  val tool = new EATool(entityDict)
  override def solve(testCase: EACase): List[String] = {
    tool.firstEntity(testCase.context :+ testCase.targetSent)
  }
}

class AffinityThreshold(entityDict: EntityDict, affinity: Affinity) extends EASolver
{
  val tool = new EATool(entityDict)
  override def solve(testCase: EACase): List[String] = {
    val candidates: List[Int] = tool.allEntityAsID(testCase.targetSent :: testCase.context).toList
    println("------------------")
    println("Target Sentence : " + testCase.targetSent)
    println("Candidates : " + (candidates map entityDict.getName).mkString(","))
    val sentenceTokens = tokenize(testCase.targetSent).toList
    val entityAffinitys: List[Double] = candidates map (affinity.top3Affinity(_, sentenceTokens))
    val candidateAnswer = candidates zip entityAffinitys
    candidateAnswer foreach { x =>
      println(entityDict.getName(x._1) + "\t: " + x._2)
    }
    val threshold = 2
    val answer = candidateAnswer filter (_._2 > threshold)
    answer map (x => entityDict.getName(x._1) )
  }
}

class ContextModel(entityDict: EntityDict)
{
  val maxContextLen = 10
  // we will see -3 ~ +3 from context
  val contextSize = 3
  def entityScore(contexts : Seq[String], targetTokens:Set[String]) : Double = {
    entityScore(contexts.slice(0, maxContextLen).toSet, targetTokens)
  }

  def entityScore(contexts : Set[String], targetTokens:Set[String]) : Double = {
    val common : Set[String] = targetTokens & contexts
    common.size
  }

  def contextWords(sentence:String, entityID : EID.EntityID) : Seq[String] = {
    // check sentence has entity
    val entityExpression = entityDict.group(entityID)
    if(entityExpression exists (sentence.contains(_)))
    {
      // We need to filter out frequent, stop words.
      val tokens :Seq[KoreanToken] = tokenizeWithPos(sentence)
      val location : Int = tokens.indexWhere( token => entityExpression exists (_.contains(token.text)))
      val prev : Seq[KoreanToken] = filterStopWords(tokens.take(location))
      val next : Seq[KoreanToken] = filterStopWords(tokens.slice(location+1, tokens.length))
      (prev.takeRight(contextSize) ++ next.take(contextSize)) map (_.text)
    }
    else
      Nil
  }

  def contextWords(testCase:EACase, entityID : EID.EntityID): Set[String] =  {
    // for all sentence in testCase, extrac
    val words : Seq[Seq[String]] = testCase.context map (contextWords(_, entityID))
    // reverse words to match priority
    val orderedWord : Seq[String] = words.reverse.flatten

    def pickFirstDistint(count:Int, alreadyPick : Set[String], remain: Seq[String]) : Set[String] = {
      if(count == 0)
        alreadyPick
      else
        remain match {
          case Nil => alreadyPick
          case head::tail =>
            if( alreadyPick contains head )
              pickFirstDistint(count, alreadyPick, tail)
            else
              pickFirstDistint(count-1, alreadyPick + head, tail)
        }

    }

    pickFirstDistint(maxContextLen, Set(), orderedWord)
  }
}

class EntityContext(entityDict: EntityDict) extends EASolver {
  val tool = new EATool(entityDict)
  val contextModel = new ContextModel(entityDict)

  def debugPrint(str:String, debug:Boolean) = {
    if(debug) println(str)
  }


  def solveByContext(testCase: EACase, debug : Boolean): List[Int] = {
    val candidates: List[Int] = tool.allEntityAsID(testCase.targetSent :: testCase.context).toList
    debugPrint("------------------", debug)
    debugPrint("Target Sentence : " + testCase.targetSent, debug)
    debugPrint("Candidates : " + (candidates map entityDict.getName).mkString(","), debug)

    val contextWords : Map[EID.EntityID, Set[String]] = {
      val pairs : List[(EID.EntityID, Set[String])] = candidates map (id => (id,contextModel.contextWords(testCase, id)))
      pairs.toMap
    }
    val targetTokens = keyTokens(testCase.targetSent).toSet
    val entityScores: List[Double] = candidates map (x => contextModel.entityScore(contextWords(x), targetTokens))
    val candidateAnswer = candidates zip entityScores
    candidateAnswer foreach { x =>
      debugPrint(entityDict.getName(x._1) + "\t: " + x._2, debug)
    }
    val threshold = 1
    val answer = candidateAnswer filter (_._2 >= threshold)
    answer map (_._1 )
  }

  override def solve(testCase: EACase): List[String] = {
    val entity1 = entityDict.extractFrom(testCase.targetSent)
    lazy val entity2 = solveByContext(testCase, false) map entityDict.getName
      entity1
  }
}

class EAContextCascade(entityDict: EntityDict) extends EASolver {
  val tool = new EATool(entityDict)
  val contextModel = new ContextModel(entityDict)

  def contextText(str:String, entityID: EntityID) : Seq[String] = {
    if(entityDict.targetContain(str, entityID))
      contextModel.contextWords(str, entityID)
    else
      keyTokens(str)
  }
  def contextTexts(list:List[(String,Boolean)], entityID: EntityID) : Seq[String] = list match {
    case (text,true)::tail => {
      contextText(text, entityID) ++ contextTexts(tail, entityID)
    }
    case (text,false)::tail => contextTexts(tail, entityID)
    case Nil => Nil
  }

  def label(texts:List[String], entity:EntityID) : List[Boolean] = texts match {
    case head::tail => {
      val preLabel = label(tail, entity)
      val explitcit = entityDict.targetContain(head, entity)
      if(explitcit)
        true :: preLabel
      else
      {
        val entityContexts = contextTexts(texts zip preLabel, entity)
        val targetTokens = keyTokens(head).toSet
        if(contextModel.entityScore(entityContexts, targetTokens) >= 1)
          true :: preLabel
        else
          false :: preLabel
      }
    }
    case Nil => List()
  }

  def solveByContext(testCase: EACase, debug : Boolean): List[Int] = {
    val allText = testCase.targetSent :: testCase.context
    val candidates: List[Int] = tool.allEntityAsID(allText).toList
    val labels : List[(Int, List[Boolean])] = candidates zip (candidates map (label(allText,_)))
    val result : List[Int] = labels filter (x => x._2.head ) map (_._1)
    result
  }

  override def solve(testCase: EACase): List[String] = {
    val entity1 = entityDict.extractFrom(testCase.targetSent)
    lazy val entity2 = solveByContext(testCase, false) map entityDict.getName
    entityDict.union(entity1,entity2)
  }
}

class MESolver(entityDict: EntityDict, trainData : List[EACase]) extends EASolver {
  val classifier = new EAClassifier(entityDict)
  val trained : MaxEnt = classifier.train(trainData)
  override def solve(testCase: EACase): List[String] = {
    classifier.predict(trained, testCase)
  }
}