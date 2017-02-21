import category._
import com.twitter.penguin.korean.TwitterKoreanProcessor
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer.KoreanToken

object scf {

  class Argument(restriction: Tag, role: String) {
    def applicable(optCategory: Option[Category]): Boolean = optCategory match
    {
      case Some(category) => category exists (_ == restriction)
      case None => false
    }
    def print() = {
      println(restriction.name + "- " + role)
    }
  }

  class SCF(val head: String, val arguments: List[Argument]) {

  }

  // TODO make object
  // Rule 1 : 보다 -

  class SCFDictionary(scfs: Iterable[SCF]) {
    val headsSet = (scfs map (_.head)).toSet
    val headIndexed = (scfs map (x => (x.head, x))).toMap

    def isKnownHead(word: String): Boolean = headsSet.contains(word)

    def get(heads: Iterable[String]): Iterable[SCF] = heads map headIndexed
  }
  def isComplete(categoryInfo: List[(String, Category)])
                (dic: SCFDictionary)
                (sentence: String)(): Boolean = {
    // 1. tokenize sentence
    val tokens: Seq[String] = stringHelper.tokenize(sentence)

    // 2. find known head
    val knownHeads: Seq[String] = tokens filter dic.isKnownHead

    // 3. match SCF pattern
    val patterns: Iterable[SCF] = dic.get(knownHeads)

    // 3-1. Tag category
    def tag = tagger(categoryInfo)(_)
    val taggedTokens: TaggedTokens = tokens map (x => (x, tag(x)))

    def satisfy(tokens: TaggedTokens)(scf: SCF): Boolean = {
      // FIXME current we match argument in greedy way
      // 1. tokens should contain head
      def matchHead(token: (String, Option[Category])): Boolean = token._1 == scf.head
      val containHead: Boolean = tokens exists matchHead

      // 2. for all arg 0~ n
      def matchArgs(tokens: TaggedTokens, args: List[Argument]): Boolean = {
        args match {
          case Nil => true
          case headArg :: tailArg => {
            val categories: Seq[Option[Category]] = tokens map (x => x._2)
            lazy val fMatch: Boolean = categories exists headArg.applicable

            def getRemain(tokens: TaggedTokens): TaggedTokens = {
              if (headArg.applicable(tokens.head._2))
                tokens.tail
              else
                tokens.head +: getRemain(tokens.tail)
            }
            lazy val fMatchTail: Boolean = matchArgs(getRemain(tokens), tailArg)
            if(!fMatch) {
              print("Failed to match")
              print("Tokens : ")
              printTokenizeResult(tokens)
              print("Args(" + args.length + ") : ")
              args foreach (_.print)
            }
            fMatch && fMatchTail
          }
        }
      }
      //   2-1. match arg0
      //   2-2. remove matched token
      // 3. if all matched then okay

      containHead && matchArgs(tokens, scf.arguments)
    }
    patterns forall (satisfy(taggedTokens))
  }
}