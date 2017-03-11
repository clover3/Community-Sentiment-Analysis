package EntityAssign
/**
  * Created by user on 2017-03-07.
  */
class EAClassifier(dict: EntityDict) {
  val tool = new EATool(dict)
  def generateCandidate(testcase:EACase) : List[String] = ???
  def toInt(f:Boolean) : Int = {
    if(f) 1
    else 0
  }
  // Present Sentence : Target Sentence
  // Previous Sentence : has object(entity)
  // Nearest Sentnece : has object(entity)

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
    val lastEntity : List[String] = tool.lastMentionedEntity(testcase.context)

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

  def feature(testCase:EACase, entity: String) : List[Int] = {
    List(
      featureDistPrevSentence(testCase), // #4
      featureConsistWithPrev(testCase, entity), // #6
      featureIsFirstMentioned(testCase, entity), // #8
      featureMostFrequent(testCase, entity) // #9
    )
  }

  def train() = {

  }
  def solve(testcase :EACase) : List[String] = {
    ???
  }

}
