package sfc
import sfc.category._
import sfc.tag._
import sfc.list._
import stringHelper._

package object sfc2 {

  class Argument(restriction: Tag, role: String) {
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

  // TODO make object
  // Rule 1 : 보다 -

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

  def listMatching[A,B](aList : List[A], bList : List[B] , matcher: (A,B) => Boolean) :
    ( List[(A,B)], List[A], List[B]) =
    // Matched Pair List, Remaining A, Remaining B
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

  def unmatchedArg(tokens: TaggedTokens)(scf: SubcategorizationFrame): List[Argument] = {
    def find(tokens: TaggedTokens, args: List[Argument]): List[Argument] = {
      args match {
        case Nil => Nil
        case headArg :: tailArg => {
          val categories: Seq[Option[Category]] = tokens map (x => x._2)
          lazy val fMatch: Boolean = categories exists headArg.applicable
          val head : Option[Argument] = if (fMatch) None else Some(headArg)

          // match head with any matching token, then return remaining
          def getRemain(tokens: TaggedTokens): TaggedTokens = tokens match {
            case Nil => Nil
            case tokenHead :: tokenTail => {
              if (headArg.applicable(tokenHead._2))
                tokenTail
              else
                tokenHead +: getRemain(tokenTail)
            }
          }

          val remainingTokens = if(fMatch) getRemain(tokens)
          else tokens

          val tail : List[Argument] = find(remainingTokens, tailArg)
          head.toList ++ tail
        }
      }
    }
    find(tokens, scf.arguments)
    def matcher(token : (String, Option[Category]), arg: Argument ) : Boolean = arg.applicable(token._2)
    val (mL, aL, bL) = listMatching(tokens.toList, scf.arguments, matcher)
    bL
  }

  def allUnmatchedArg(categoryInfo: List[(String, Category)])
                (dic: SCFDictionary)
                (sentence: String) : Iterable[Argument] = {
    // 1. tokenize sentence
    val tokens: Seq[String] = stringHelper.tokenize(sentence)

    // 2. find known head
    val knownHeads: Seq[String] = tokens filter dic.isKnownHead

    // 3. match SCF pattern
    val patterns: Iterable[SubcategorizationFrame] = dic.get(knownHeads)

    // 3-1. Tag category
    def tag = tagger(categoryInfo)(_)
    val taggedTokens: TaggedTokens = tokens map (x => (x, tag(x)))

    patterns flatMap (unmatchedArg(taggedTokens))
  }

  def applyPossibleSCF(categoryInfo: List[(String, Category)])
                      (dic: SCFDictionary)
                      (sentence: String) : Iterable[Argument] = {
    val tokens: Seq[String] = stringHelper.tokenize(sentence)

    // 2. find known head
    val knownHeads: Seq[String] = tokens filter dic.isKnownHead

    // 3. match SCF pattern
    val patterns: Iterable[SubcategorizationFrame] = dic.get(knownHeads)

    // 3-1. Tag category
    def tag = tagger(categoryInfo)(_)
    val taggedTokens: TaggedTokens = tokens map (x => (x, tag(x)))
    def matcher(token : (String, Option[Category]), arg: Argument ) : Boolean = arg.applicable(token._2)
    def f = listMatching(taggedTokens.toList)(_)(matcher)
    patterns map f
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


}