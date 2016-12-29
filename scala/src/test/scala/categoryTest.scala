
import org.scalatest.FunSuite
import category._

class CategorySuite extends FunSuite {

  test("check category load"){
    loadCategory("data\\StdDic.txt", "data\\Synset.txt", "data\\Tran.txt")
  }
}
