
import org.scalatest.FunSuite
import category._

class CategorySuite extends FunSuite {

  test("check category load"){
    val categories = loadCategory("data\\StdDic.txt", "data\\Synset.txt", "data\\Tran.txt")

    def printer(entry : (String, Category)) = {
      print(entry._1)
      print(":")
      printCategory(entry._2)
      println("")
    }
    val mother = categories filter ( x=> x._1 contains "차")
    mother foreach printer

  }
}
