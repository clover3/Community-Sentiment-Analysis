package EntityAssign
import maxent.{Instance, MaxEnt}
import stringHelper._

import scala.collection.JavaConversions._
/**
  * Created by user on 2017-03-07.
  */
class EAClassifier(dict: EntityDict) {
  val LABEL_TRUE = 1
  val LABEL_FALSE = 2
  val tool = new EATool(dict)
  def generateCandidate(testcase:EACase) : List[String] = ???
  def toInt(f:Boolean) : Int = {
    if(f) 1
    else 0
  }
  // Present Sentence : Target Sentence
  // Previous Sentence : has object(entity)
  // Nearest Sentnece : has object(entity)

  // feature #1
  def featureHasEntity(testcase: EACase) : Int = {
    val entity : Option[String] = dict.extractAnyFrom(testcase.targetSent)
    entity match {
      case Some(s) => 1
      case None => 0
    }
  }

  // feature #2
  def featureComparative(testcase:EACase) : Int = {
    val compKeywords : List[String] = List("보다","비하면", "비교해서", "비슷", "한 수 위", "보단", "차라리", "나음")
    def containAny(str: String, words : List[String]) : Boolean = {
      words exists (str.contains(_))
    }
    if( containAny(testcase.targetSent, compKeywords) )
      1
    else tool.prevSentence(testcase.context) match {
      case None => 0
      case Some(s) => toInt(containAny(s, compKeywords))
    }
  }

  // feature #3

  // feature #4
  def featureDistPrevSentence(testcase: EACase) : Int = {
    val dist = testcase.context.reverse.indexWhere(dict.hasAny(_))
    if(dist < 0)
      -1
    else
      dist +1
  }

  // feature #6
  def featureConsistWithPrev(testcase: EACase, entity : String) : Int = {
    val lastEntity : List[String] = tool.lastMentionedEntitys(testcase.context)

    val fConsistent = (lastEntity map dict.getGroup).toSet.contains(dict.getGroup(entity))
    toInt(fConsistent)
  }

  // feature #8
  def featureIsFirstMentioned(testcase: EACase, entity : String) : Int = {
    val entitys : List[String] = tool.firstEntity(testcase.context)
    val f = (entitys map dict.getGroup).toSet.contains(dict.getGroup(entity))
    toInt(f)
  }

  // feature #9
  def featureMostFrequent(testcase: EACase, entity : String) : Int = {
    val allText : List[String] = testcase.targetSent::testcase.context
    toInt(tool.isMostFrequent(allText, entity))
  }

  def featureCurrent(testcase: EACase, entity : String) : Int = {
    val curEntity : List[String] = dict.extractFrom(testcase.targetSent)
    val fHas = curEntity.toSet.contains(entity)
    toInt(fHas)
  }


  def feature(testCase:EACase, entity: String) : Array[Int] = {
    Array(
      featureHasEntity(testCase), // #1
      featureComparative(testCase), // #2
      // #3 is not described
      featureDistPrevSentence(testCase), // #4
      // #5 requires next text
      featureConsistWithPrev(testCase, entity), // #6
      // #7 requires next text
      featureIsFirstMentioned(testCase, entity), // #8
      featureMostFrequent(testCase, entity), // #9
      featureCurrent(testCase,entity)
    )
  }
  def conver2test(testcase: EACase, entity : String) : Instance = {
    val instance = new Instance(3, feature(testcase, entity))
    instance
  }
  def train(data : List[EACase]) : MaxEnt = {
    def convert2Instances(testcase : EACase) : List[Instance] = {
      val candidates : Set[String] = tool.allEntity(testcase.targetSent::testcase.context)
      val wrongCandidates : Iterable[String] = dict.exclusive(candidates, testcase.entity)
      val explicit : Iterable[String] = dict.extractFrom(testcase.targetSent)
      val implcitEntity = dict.exclusive(testcase.entity, explicit)
      val nonimplicitEntity = dict.exclusive(candidates, implcitEntity)

      val trueCase : Iterable[Instance] = for(
        entity <- implcitEntity
      ) yield new Instance(LABEL_TRUE, feature(testcase, entity))

      val falseCases : Iterable[Instance] = for(
        entity <- nonimplicitEntity
      ) yield new Instance(LABEL_FALSE, feature(testcase, entity))
      trueCase.toList ++ falseCases
    }

    val trainData : List[Instance] = data flatMap convert2Instances
    val me: MaxEnt = new MaxEnt(trainData)
    me.train()
    me
  }
  def predict(trained : MaxEnt, testcase :EACase) : List[String] = {
    val candidate : Set[String] = tool.allEntity(testcase.targetSent::testcase.context)
    def test(x: String) : Boolean = trained.classify(conver2test(testcase, x)) == LABEL_TRUE
    val implicitEntity = (candidate filter test).toList
    val explicit = dict.extractFrom(testcase.targetSent)
    dict.union(explicit, implicitEntity)
  }

}
