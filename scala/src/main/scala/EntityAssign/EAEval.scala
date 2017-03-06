package EntityAssign

import java.io.{BufferedReader, File, FileInputStream, InputStreamReader}
import java.lang.NumberFormatException

/**
  * Created by user on 2017-02-28.
  */

// Sent = Sentence

/*
   Dict Format
   [Entity ID#1]\t[EntityString#1]\t[EntityString#2]
   [Entity ID#2]\t[EntityString#3]\t[EntityString#4]\t[EntityString#5]
   ... So on
 */
class EntityDict(dictPath : String)
{
  val entityInfo : List[(String, Int)] = {
    val itr = io.Source.fromFile(dictPath).getLines
    val lines: List[String] = itr.toList
    def parseLine(line: String): List[(String, Int)] = {
      try {
        //val idx = line.indexOf(' ')
        //val groupNum = line.substring(0,idx).toInt
        //val entity = line.substring(idx+1).trim
        val tokens = line.trim().split("\t")
        val groupNum = tokens(0).trim().toInt
        val entitys = tokens.slice(1, tokens.length).toList
        entitys map (x => (x, groupNum))
      }catch{
        case e:NumberFormatException => throw e
      }
    }
    (lines map parseLine) flatten
  }
  val entityList : List[String] = entityInfo map (_._1)
  val entity2group : Map[String, Int]= entityInfo.toMap
  val group : Map[Int, List[String]] = {
    val groupedTemp = entityInfo.groupBy(_._2)
    groupedTemp map (x => (x._1, x._2 map (_._1)))
  }
  def extractFrom(str : String) : List[String] = {
    def getIfExist(dest: String)(pattern: String) : Option[String] = {
      val idx = dest.toLowerCase().indexOfSlice(pattern.toLowerCase())
      if(idx < 0)
        None
      else if(idx == 0)
        Some(pattern)
      else {
        val preChar = dest(idx-1)
        if(List('.',' ' ,'\n' ,'?').contains(preChar))
          Some(pattern)
        else
          None
      }
    }
    (entityList map getIfExist(str)) flatten
  }
  def extractAnyFrom(str : String) : Option[String] = {
    val r = extractFrom(str)
    if( r.isEmpty)
      None
    else
      Some(r.head)
  }
}

class EACase(val entity : String, val targetSent : String, val context : List[String]){

}

class EAEval(dirPath : String, entityDict: EntityDict) {
  def readEuckr(file:File) : Stream[String] = {
    val br: BufferedReader =  new BufferedReader(new InputStreamReader(new FileInputStream(file),"euc-kr"))
    val strs: Stream[String] = Stream.continually(br.readLine()).takeWhile(_ != null)
    strs
  }

  def parseContextSentences(lines: Array[String]): List[String] = {
    if (lines.length == 0)
      return Nil
    else {
      val strContextLen = lines(0).toInt
      val context = lines.slice(1, 1 + strContextLen).mkString("\n")
      context :: parseContextSentences(lines.slice(1 + strContextLen, lines.length))
    }
  }

  def loadCase(file: File): EACase = {
    try {
      val lines: Array[String] = readEuckr(file).toArray
      val entity = lines(0)
      val strTargetLen = lines(1).toInt
      val strTarget = lines.slice(2, 2 + strTargetLen).mkString("\n")

      val contexts: List[String] = parseContextSentences(lines.slice(2 + strTargetLen, lines.length))
      new EACase(entity, strTarget, contexts)
    }catch {
      case e: NumberFormatException => {
        println(file.getPath)
        throw e
      }
    }

  }
  val testCases : List[EACase] = {
    // Enum Dir
    val files : List[File] = {
      val d = new File(dirPath)
      if (d.exists && d.isDirectory) {
        d.listFiles.filter(_.isFile).toList
      } else {
        List[File]()
      }
    }
    val cases  : List[EACase] = files map loadCase
    cases
  }

  def evalPerformance(solver :EASolver) : Float = {
    val results : List[Option[String]] = testCases map solver.solve
    val total  = results.length
    def isSuccess(arg : (EACase, Option[String])) : Boolean = {
      val answer = arg._1.entity
      val found = arg._2
      found match {
        case None => answer == "-"
        case Some(s) => {
          if( answer == "-")
            false
          else
            entityDict.entity2group(s) == entityDict.entity2group(answer)
        }
      }
    }
    val suc :Int = (testCases zip results) count isSuccess
    return (suc.toFloat/total)
  }

  def showResult(solver : EASolver) = {
    val results : List[Option[String]] = testCases map solver.solve
    def show(item : (Option[String], EACase)) : Unit= {
      val sentence = item._2.targetSent
      val answer = item._2.entity
      val strResult = item._1 match {
        case None => "None"
        case Some(s) => s
      }
      if( strResult == answer)
        println(s"$sentence : $strResult(Correct)")
      else
        println(s"$sentence : $strResult($answer)")
    }
    (results zip testCases) foreach show
  }
}

trait EASolver {
  def solve(testCase : EACase) : Option[String]
}

class RecentFirst(entityDict: EntityDict) extends EASolver {
  override def solve(testCase: EACase): Option[String] = {
    val entityOfTarget : Option[String] = entityDict.extractAnyFrom(testCase.targetSent)
    def lastMentionedEntity(contexts : List[String]) : Option[String] = {
      // If tail has Some give it, if tail None, then do at the head
      contexts match {
        case Nil => None
        case head::tail => lastMentionedEntity(tail) match {
          case Some(x) => Some(x)
          case None => entityDict.extractAnyFrom(head)
        }
      }
    }
    if (entityOfTarget.isDefined)
      entityOfTarget
    else{
      lastMentionedEntity(testCase.context)
    }
  }
}

class FirstOnly(entityDict: EntityDict) extends EASolver {
  def firstEntity(texts : List[String]) : Option[String] = {
    texts match {
      case Nil => None
      case head::tail => entityDict.extractAnyFrom(head) match {
        case None => firstEntity(tail)
        case Some(s) => Some(s)
      }
    }

  }
  override def solve(testCase: EACase): Option[String] = {
    firstEntity(testCase.context :+ testCase.targetSent)
  }
}