import java.util
import java.util.regex.Pattern

import org.scalatest.FunSuite
import cc.mallet.classify.{Classifier, ClassifierTrainer, MaxEntTrainer, Trial}
import cc.mallet.pipe._
import cc.mallet.types._
import cc.mallet.util.Randoms
/**
  * Created by user on 2017-03-07.
  */
class MalletTest extends FunSuite {

  test("Mallet test") {
    val labelIndex = 3
    def createToken() : Token = {
      val token : Token= new Token("exampleToken");

      // Note: properties are not used for computing (I think)
      token.setProperty("SOME_PROPERTY", "hello");

      // Any old double value
      token.setFeatureValue("F1", 666.0);
      token
    }
    val labelAlphabet = new LabelAlphabet();
    val observations : TokenSequence = new TokenSequence();
    val labels : LabelSequence = new LabelSequence(labelAlphabet, 10);

    observations.add(createToken());
    labels.add("idk, some target or something");

    new Instance(
      observations,
      labels.getLabelAtPosition(labelIndex),
      "myInstance",
      null
    );

  }
}
