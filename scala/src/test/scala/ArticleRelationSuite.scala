import org.scalatest.FunSuite

/**
  * Created by user on 2017-03-14.
  */
class ArticleRelationSuite extends FunSuite {
  test("A Case")
  {
    val rel = new ArticleRelation("..\\input\\bobae_relation.txt")
    assert(rel(1127725,211873) === (1127725,1127725))
  }
}
