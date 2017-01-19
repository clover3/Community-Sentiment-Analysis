
import kaist.cilab.parser.berkeleyadaptation.BerkeleyParserWrapper

/**
  * Created by user on 2017-01-18.
  */
object KaistParser {
  val parserPath: String = "./models/parser/KorGrammar_BerkF_FIN"
  val bpw: BerkeleyParserWrapper = new BerkeleyParserWrapper(parserPath) // Path for parser model


  type TokenType = String

  class Node(tokenType: TokenType, str : String, childs : List[Node]){
    def allStr : String =
      if(childs == Nil) str
      else (childs.map (_.allStr)).mkString("")
    def allTags : String =
      if(childs == Nil) tokenType + " " + str
      else (childs.map (_.allTags)).mkString("")
  }

  class ParseTree(val root: Node)


  def parse(rawSentence : String) : ParseTree = {
    val replaceParen = rawSentence.contains('(') || rawSentence.contains(')')

    def removeCharsInsideParen(str:String) : String = str.replaceAll("\\(.*?\\)", "")

    val inputSentence = if(replaceParen) removeCharsInsideParen(rawSentence)
    else rawSentence

    val treeString : String = bpw.parse(inputSentence)

    def subParse(str: String) : Node = {
      val len = str.length
      assert(str(0).equals('(') && str(len - 1).equals(')'))
      val inner = str.substring(1, len - 1)
      val tokenType = inner.split(" ")(0)
      val remainStr = inner.substring(tokenType.length)

      def horizontalParse(str: String): List[Node] = {
        def closeParenIdx(strRemain: String, openIdx: Int): Int = {
          var idx = openIdx + 1
          var parenOpen = 1
          while (idx < strRemain.length && parenOpen > 0) {
            if (strRemain(idx) == '(')
              parenOpen += 1
            else if (strRemain(idx) == ')')
              parenOpen -= 1
            idx += 1
          }
          idx - 1
        }
        val idxFirstParenOpen = str.indexOf('(')
        val idxFirstParenClose = closeParenIdx(str, idxFirstParenOpen)

        val node: Node = subParse(str.substring(idxFirstParenOpen, idxFirstParenClose + 1))
        val right: List[Node] = {
          try {
            val remain = str.substring(idxFirstParenClose + 1)
            horizontalParse(remain)
          }
          catch {
            case e: StringIndexOutOfBoundsException => Nil
          }
        }
        node :: right
      }

      if (inner.contains('(')) {
        val childs = horizontalParse(remainStr)
        new Node(tokenType, "-", childs)
      }
      else
        new Node(tokenType, remainStr, Nil)
    }
    val start = treeString.indexOf("(")
    val end = treeString.lastIndexOf(")")
    val root = subParse( treeString.substring(start,end+1) )
    new ParseTree(root)
  }

}
