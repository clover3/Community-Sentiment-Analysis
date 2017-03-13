package EntityAssign

/**
  * Created by user on 2017-03-10.
  */
class Affinity(path : String, entityDict: EntityDict) {
  def load() = {
    val itr = io.Source.fromFile(path).getLines
    val lines: List[String] = itr.toList
    def parseLine(line: String): (String, Seq[String], Double) = {
      try {
        val tokens = line.split("\t")
        assert( tokens.length == 3)

        val entity = tokens(0)
        val words : Seq[String] = tokens(1).split(",")
        val affinity = tokens(2).toDouble
        (entity, words, affinity)
      } catch {
        case e: Exception => println(line)
          throw e
      }
    }

    // List( EntityID, texts, affinity)
    val rawData : List[(Int, Seq[String], Double)] = lines map parseLine map (x => (entityDict.getGroup(x._1), x._2, x._3))

    // all items have same Int(EntityID) value, it would be ignored
    def convert(item : List[(Int, Seq[String], Double)]) : Map[String, Double] = {
      val pairs : List[Seq[(String, Double)]] = item map (x => x._2 map {y => (y,x._3)} )
      pairs.flatten.toMap
    }
    val tempGroup : Map[Int, Map[String, Double]] = rawData.groupBy (_._1) mapValues(convert)
    tempGroup
  }

  val data : Map[Int, Map[String, Double]] = load()
  def get(entityID:EID.EntityID, word : String) : Double = {
    if(data .contains(entityID))
    {
      val affinitys : Map[String, Double] = data(entityID)
      if( affinitys.contains(word) )
        affinitys(word)
      else
        1
    }
    else
      1
  }

  def get(entity: String, word : String) : Double = {
    val entityID = entityDict.getGroup(entity)
    get(entityID, word)
  }

  def averageAffinity(entityID : EID.EntityID, words : List[String]) : Double = {
    val affs : List[Double] = words map (get(entityID, _))
    val sums : Double = affs.foldRight(0.0)((x,y) => x+y)
    sums / words.length
  }

  def top3Affinity(entityID : EID.EntityID, words : List[String]) : Double = {
    val affs : List[Double] = words map (get(entityID, _))
    val focus = affs.sortWith(_ > _).take(3)

    val sums : Double = focus.foldRight(0.0)((x,y) => x+y)
    sums / focus.length
  }

  def averageAffinity(entity : String, words : List[String]) : Double = {
    val entityID = entityDict.getGroup(entity)
    averageAffinity(entityID, words)
  }

}
