/**
  * Created by user on 2017-03-14.
  */
class ArticleRelation(path:String) {
  type ArticleID = Int
  type ThreadID = Int
  type TAID = (ThreadID, ArticleID)
  private val data : Map[TAID,TAID] = {
      val itr = io.Source.fromFile(path).getLines
      val lines: List[String] = itr.toList
      def parseLine(line:String) : ((Int,Int),(Int,Int)) = {
        val tokens = line.split(",")
        assert(tokens.length == 4)
        val iTokens : Array[Int] = tokens map (_.toInt)
        ((iTokens(0),iTokens(1)),(iTokens(2),iTokens(3)))
      }
      (lines map parseLine).toMap
    }
  def apply(atID: TAID) : TAID = data(atID)
}
