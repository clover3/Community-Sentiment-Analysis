import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken
import com.twitter.penguin.korean.util.KoreanPos

/**
  * Created by user on 2017-02-21.
  */
package object stringHelper {
  def tokenize(text : String) : Seq[String] =
  {
    val normalized: CharSequence = TwitterKoreanProcessor.normalize(text)
    val tokens: Seq[KoreanToken] = TwitterKoreanProcessor.tokenize(normalized)
    val stemmed: Seq[KoreanToken] = TwitterKoreanProcessor.stem(tokens)

    stemmed filterNot (_.pos == KoreanPos.Josa) map (x => x.text)
  }

}
