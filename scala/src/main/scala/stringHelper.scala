import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken
import com.twitter.penguin.korean.util.KoreanPos
import com.twitter.penguin.korean.util.KoreanPos.KoreanPos

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
  def tokenizeWithPos(text : String) : Seq[KoreanToken] =
  {
    val normalized: CharSequence = TwitterKoreanProcessor.normalize(text)
    val tokens: Seq[KoreanToken] = TwitterKoreanProcessor.tokenize(normalized)
    val stemmed: Seq[KoreanToken] = TwitterKoreanProcessor.stem(tokens)

    stemmed filterNot (_.pos == KoreanPos.Josa)
  }

  def filterStopWords(tokens : Seq[KoreanToken]):Seq[KoreanToken] = {
    val notImportant = List( KoreanPos.Josa, KoreanPos.Suffix, KoreanPos.Eomi, KoreanPos.Exclamation, KoreanPos.KoreanParticle)
    val important = List(KoreanPos.Noun, KoreanPos.Verb, KoreanPos.ProperNoun, KoreanPos.Alpha, KoreanPos.Adjective, KoreanPos.Adverb)
    val res = tokens filter {x => (important.contains(x.pos)) }
    res
  }

  def keyTokens(text : String) : Seq[String] =
  {
    val normalized: CharSequence = TwitterKoreanProcessor.normalize(text)
    val tokens: Seq[KoreanToken] = TwitterKoreanProcessor.tokenize(normalized)
    val stemmed: Seq[KoreanToken] = TwitterKoreanProcessor.stem(tokens)

    filterStopWords(stemmed) map (x => x.text)
  }

}
