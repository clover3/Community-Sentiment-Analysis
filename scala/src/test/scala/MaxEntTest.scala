import java.io.FileNotFoundException

import org.scalatest.FunSuite

import scala.collection.JavaConversions._
import java.util.List
import maxent.Instance
import maxent.{DataSet, MaxEnt};


/**
  * Created by user on 2017-03-07.
  */
class MaxEntTest extends FunSuite {

  test("example") {
    val instances: List[Instance] = DataSet.readDataSet("examples/zoo.train")
    val me: MaxEnt = new MaxEnt(instances)
    me.train()
    val trainInstances: List[Instance] = DataSet.readDataSet("examples/zoo.test")
    var pass: Int = 0
    for(instance <- trainInstances) {
      val predict: Int = me.classify(instance)
      if (predict == instance.getLabel) pass += 1
    }
    System.out.println("accuracy: " + 1.0 * pass / trainInstances.size)
  }

}
