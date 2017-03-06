package sfc
import sfc.category._
import sfc.tag._
import sfc.list._
import stringHelper._

package object sfc2 {

  class Argument(restriction: Tag, val role: String) {
    def applicable(optCategory: Option[Category]): Boolean = optCategory match
    {
      case Some(category) => category exists (_ == restriction)
      case None => false
    }
    def print() = {
      val tag = restriction.name
      println(s"Arg[$role] $tag")
    }
  }

  class SubcategorizationFrame(val head: String, val arguments: List[Argument]) {  }

  class SCFDictionary(scfs: Iterable[SubcategorizationFrame]) {
    val headsSet = (scfs map (_.head)).toSet
    val headIndexed = (scfs map (x => (x.head, x))).toMap

    def isKnownHead(word: String): Boolean = headsSet.contains(word)

    def get(heads: Iterable[String]): Iterable[SubcategorizationFrame] = heads map headIndexed
  }

  def scfDictBest(tags :Set[tag.Tag]) : SCFDictionary = {
    val generator = new Generator(tags)

    new SCFDictionary(List(
      generator.ride,
      generator.see,
      generator.beingpretty,
      generator.beingpretty2,
      generator.buy,
      generator.buy2
    ))
  }

  // Return : (Matched Pair List, Remaining A, Remaining B)
  def listMatching[A,B](aList : List[A], bList : List[B] , matcher: (A,B) => Boolean) :
    ( List[(A,B)], List[A], List[B]) =
  {
      def getFirstMatch(aList: Iterable[A], b : B) : Option[A] = aList find (matcher(_, b))
      def filterFrom(aList : List[A], optA : Option[A]) : List[A] = optA match {
        case None => aList
        case Some(a) => aList.filterNot( _ == a)
      }
      bList match {
        case Nil => (Nil, aList, Nil)
        case bHead :: bTail =>
        {
          val found : Option[A] = getFirstMatch(aList, bHead)
          val aTail: List[A] = filterFrom(aList, found)
          val (mL, aL, bL) = listMatching(aTail, bTail, matcher)

          found match {
            case Some(a) =>( (a,bHead):: mL , aL, bL)
            case None => (mL, aL, bHead::bL)
          }
        }
      }
    }

  // This function is unused
  def unmatchedArg(tokens: TaggedTokens)(scf: SubcategorizationFrame): List[Argument] = {
    def matcher(token : (String, Option[Category]), arg: Argument ) : Boolean = arg.applicable(token._2)
    val (mL, aL, bL) = listMatching(tokens.toList, scf.arguments, matcher)
    bL
  }


  def allUnmatchedArg(categoryInfo: List[(String, Category)])
                (dic: SCFDictionary)
                (sentence: String) : Iterable[Argument] = {
    val SCFMatchResults = applyPossibleSCF(categoryInfo)(dic)(sentence)
    SCFMatchResults map (_._4) flatten
  }

  type SCFMatch = (SubcategorizationFrame, List[(TaggedToken, Argument)], List[TaggedToken], List[Argument])
  implicit class SCFMatchCompanionOps(val s: SCFMatch) extends AnyVal {
    def print ={
      def matchPrinter(x:(TaggedToken, Argument)) = {
        val str : String = x._1._1
        val arg : String = x._2.role
        println(s"$str - $arg")
      }

      s match { case(scf, mL, tokenL, argL) => {
          val head = scf.head
          println(s"[SCF] head = $head")
          mL foreach matchPrinter
          argL foreach (arg => Predef.println("??? - " + arg.role))
        }
      }
    }
  }

  // Given sentence, tokenize the sentence, tag the sentence and match it with possible SCF
  def applyPossibleSCF(categoryInfo: List[(String, Category)])
                      (dic: SCFDictionary)
                      (sentence: String) : Iterable[SCFMatch] = {
    val tokens: Seq[String] = stringHelper.tokenize(sentence)

    // 2. find known head
    val knownHeads: Seq[String] = tokens filter dic.isKnownHead

    // 3. match SCF pattern
    val patterns: Iterable[SubcategorizationFrame] = dic.get(knownHeads)

    // 3-1. Tag category
    def tag = tagger(categoryInfo)(_)
    val taggedTokens: TaggedTokens = tokens map (x => (x, tag(x)))
    def matcher(token : (String, Option[Category]), arg: Argument ) : Boolean = arg.applicable(token._2)
    def apply(x:SubcategorizationFrame) = listMatching(taggedTokens.toList, x.arguments, matcher)
    patterns map ( scf => ( apply(scf) match {case(mL,aL,bL) => (scf, mL, aL, bL ) } ))
  }

  def isComplete(categoryInfo: List[(String, Category)])
                (dic: SCFDictionary)
                (sentence: String)(): Boolean = {
    // 1. tokenize sentence
    val tokens: Seq[String] = stringHelper.tokenize(sentence)

    // 2. find known head
    val knownHeads: Seq[String] = tokens filter dic.isKnownHead

    // 3. match SCF pattern
    val patterns: Iterable[SubcategorizationFrame] = dic.get(knownHeads)

    // 3-1. Tag category
    def tag = tagger(categoryInfo)(_)
    val taggedTokens: TaggedTokens = tokens map (x => (x, tag(x)))

    def satisfy(tokens: TaggedTokens)(scf: SubcategorizationFrame): Boolean = {
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

            def getRemain(tokens: TaggedTokens): TaggedTokens = tokens match {
              case Nil => Nil
              case tokenHead :: tokenTail => {
                if (headArg.applicable(tokenHead._2))
                  tokenTail
                else
                  tokenHead +: getRemain(tokenTail)
              }
            }

            lazy val fMatchTail: Boolean = matchArgs(getRemain(tokens), tailArg)

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

  class Ceylon(categoryInfo: List[(String, Category)], dic: SCFDictionary)
  {
    def recover(target: String, context: String): List[String] = {
      val remainArg : Iterable[Argument] = allUnmatchedArg(categoryInfo)(dic)(target)
      val tokens: Seq[String] = stringHelper.tokenize(context)
      def tag = tagger(categoryInfo)(_)
      val taggedTokens: TaggedTokens = tokens map (x => (x, tag(x)))

      def matcher(token : (String, Option[Category]), arg: Argument ) : Boolean = arg.applicable(token._2)
      val (mL, tokenL, argL) = listMatching(taggedTokens.toList, remainArg.toList, matcher)
      mL map (_._1._1)
    }

    def showRecovery(text: String, textContext: String) = {
      val recovered = recover(text, textContext)
      val len = recovered.length
      if (len > 0) {
        println(s"Context: $textContext")
        println(s"Text: $text")
        val outOmit = recovered.mkString(",")
        print(s"Omitted($len) : $outOmit")
      }
    }
  }
}