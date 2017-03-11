package EntityAssign

/**
  * Created by user on 2017-03-10.
  */
class Affinity(path : String, entityDict: EntityDict) {
  def load() = {
    val itr = io.Source.fromFile(path).getLines
    val lines: List[String] = itr.toList
    def parseLine(line: String): (String, Seq[String], Float) = {
      try {
        val tokens = line.split("\t")
        assert( tokens.length == 3)

        val entity = tokens(0)
        val words : Seq[String] = tokens(1).split(",")
        val affinity = tokens(2).toFloat
        (entity, words, affinity)
      } catch {
        case e: Exception => println(line)
          throw e
      }
    }

    // List( EntityID, texts, affinity)
    val rawData : List[(Int, Seq[String], Float)] = lines map parseLine map (x => (entityDict.getGroup(x._1), x._2, x._3))

    // all items have same Int(EntityID) value, it would be ignored
    def convert(item : List[(Int, Seq[String], Float)]) : Map[String, Float] = {
      val pairs : List[Seq[(String, Float)]] = item map (x => x._2 map {y => (y,x._3)} )
      pairs.flatten.toMap
    }
    val tempGroup : Map[Int, Map[String, Float]] = rawData.groupBy (_._1) mapValues(convert)
    tempGroup
  }

  val data : Map[Int, Map[String, Float]] = load()

  def get(entity: String, word : String) : Float = {
    val entityID = entityDict.getGroup(entity)
    val affinitys : Map[String, Float] = data(entityID)
    if( affinitys.contains(word) )
      affinitys(word)
    else
      1
  }
}
