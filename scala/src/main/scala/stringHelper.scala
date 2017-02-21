import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken

/**
  * Created by user on 2017-02-21.
  */
object stringHelper {
  def tokenize(text : String) : Seq[String] =
  {
    val normalized: CharSequence = TwitterKoreanProcessor.normalize(text)
    val tokens: Seq[KoreanToken] = TwitterKoreanProcessor.tokenize(normalized)
    val stemmed: Seq[KoreanToken] = TwitterKoreanProcessor.stem(tokens)
    stemmed map (x => x.text)
  }

}