import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken
import com.twitter.penguin.korean.util.KoreanPos
import com.twitter.penguin.korean.util.KoreanPos.KoreanPos

/**
  * Created by user on 2017-02-21.
  */
package object stringHelper {
  val cache : scala.collection.mutable.Map[String, Seq[KoreanToken]] = scala.collection.mutable.Map()

  private def stem(text : String) : Seq[KoreanToken] = {
    if(cache.contains(text))
    {
      cache(text)
    }
    else {
      val normalized: CharSequence = TwitterKoreanProcessor.normalize(text)
      val tokens: Seq[KoreanToken] = TwitterKoreanProcessor.tokenize(normalized)
      val stemmed: Seq[KoreanToken] = TwitterKoreanProcessor.stem(tokens)
      val cache2 = cache+= (text->stemmed)
      stemmed
    }
  }

  def tokenize(text : String) : Seq[String] =
  {
    stem(text) filterNot (_.pos == KoreanPos.Josa) map (x => x.text)
  }
  def tokenizeWithPos(text : String) : Seq[KoreanToken] =
  {
    stem(text) filterNot (_.pos == KoreanPos.Josa)
  }

  def filterStopWords(tokens : Seq[KoreanToken]):Seq[KoreanToken] = {
    val notImportant = List( KoreanPos.Josa, KoreanPos.Suffix, KoreanPos.Eomi, KoreanPos.Exclamation, KoreanPos.KoreanParticle)
    val important = List(KoreanPos.Noun, KoreanPos.Verb, KoreanPos.ProperNoun, KoreanPos.Alpha, KoreanPos.Adjective, KoreanPos.Adverb)
    val res = tokens filter {x => (important.contains(x.pos)) }
    res
  }

  def keyTokens(text : String) : Seq[String] =
  {
    filterStopWords(stem(text)) map (x => x.text)
  }

}
