package EntityAssign

import stringHelper.keyTokens
import java.io.{BufferedReader, File, FileInputStream, InputStreamReader}
import java.lang.NumberFormatException

import EntityAssign.EID.EntityID

import scala.collection.immutable.Stream.Empty

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
object EID {
  type EntityID = Int
}

class EntityDict(dictPath : String)
{

  val entityInfo : List[(String, EntityID)] = {
    val itr = io.Source.fromFile(dictPath).getLines
    val lines: List[String] = itr.toList
    def parseLine(line: String): List[(String, Int)] = {
      try {
        //val idx = line.indexOf(' ')
        //val groupNum = line.substring(0,idx).toInt
        //val entity = line.substring(idx+1).trim
        val tokens = line.trim().split("\t")
        val groupNum = tokens(0).trim().toInt
        val entitys = tokens.slice(1, tokens.length).toList map (_.toLowerCase)
        val nonEmpty = (entitys filterNot (_.length==0))
        nonEmpty map (x => (x, groupNum))
      } catch {
        case e:NumberFormatException => throw e
      }
    }
    (lines map parseLine) flatten
  }
  // List of all entity
  val entityList : List[String] = entityInfo map (_._1)
  private val entity2group : Map[String, Int]= entityInfo.toMap
  val group : Map[Int, List[String]] = {
    val groupedTemp = entityInfo.groupBy(_._2)
    groupedTemp map (x => (x._1, x._2 map (_._1)))
  }
  val cache : scala.collection.mutable.Map[String, List[String]] = scala.collection.mutable.Map()

  def getName(eid : EID.EntityID) : String = group(eid).head
  def getGroup(entity: String) : Int = entity2group(entity.toLowerCase)
  def has(entity:String) : Boolean = entity2group.contains(entity.toLowerCase)
  def hasAny(str:String) : Boolean = extractFrom(str) != Nil

  def startOfToken(c : Char) : Boolean = List('.',' ' ,'\n' ,'?').contains(c)
  def isAlphabet(c:Char) : Boolean = c.isLower || c.isUpper
  def langInversion(c:Char, c2:Char) : Boolean = !isAlphabet(c) && isAlphabet(c2)

  def extractFrom(str : String) : List[String] = {
    if( cache.contains(str) ) {
      cache(str)
    }
    else
    {
      def getIfExist(dest: String)(pattern: String) : Option[String] = {
        val idx = dest.toLowerCase().indexOfSlice(pattern.toLowerCase())
        if(idx < 0)
          None
        else if(idx == 0)
          Some(pattern)
        else if(startOfToken(dest(idx-1)))
          Some(pattern)
        else if(langInversion(dest(idx-1), dest(idx)))
          Some(pattern)
        else
          getIfExist(dest.substring(idx+1))(pattern)
      }
      val temp : List[Option[String]] = entityList map getIfExist(str)
      val result : List[String] = temp flatten
      val cache2 = cache+= (str->result)
      result
    }
  }
  def extractAnyFrom(str : String) : Option[String] = {
    val r = extractFrom(str)
    if( r.isEmpty)
      None
    else
      Some(r.head)
  }

  def targetContain(str:String, entity :EntityID) : Boolean = {
    val r = extractFrom(str).toSet map getGroup
    r.contains(entity)
  }

  // A-B
  def exclusive(setA:Iterable[String], setB:Iterable[String]) : List[String] = {
    val result : Set[Int] = (setA map getGroup toSet) -- (setB map getGroup)
    result.toList map getName
  }
  def union(setA:Iterable[String], setB:Iterable[String]) : List[String] = {
    val result : Set[Int] = (setA map getGroup toSet) ++ (setB map getGroup)
    result.toList map getName
  }
}



// context : from top->bottom, old->recent
class EACase(val entity : Iterable[String], val targetSent : String, val context : List[String]){

}

class EAEval(dirPath : String, entityDict: EntityDict) {
  val tool = new EATool(entityDict)
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
      val rawThread = lines(0)
      val rawEntity = lines(1)
      val entitys:Iterable[String] = {
        if(rawEntity== "-")
          Nil
        else
          rawEntity.split(",") map (_.trim)
      }

      val strTargetLen = lines(2).toInt
      val strTarget = lines.slice(3, 3 + strTargetLen).mkString("\n")

      val contexts: List[String] = parseContextSentences(lines.slice(3 + strTargetLen, lines.length))
      new EACase(entitys, strTarget, contexts)
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

  def isSuccess(arg : (EACase, List[String])) : Boolean = {
    val answer : Iterable[Int] = arg._1.entity map entityDict.getGroup
    val found : Iterable[Int] = arg._2 map entityDict.getGroup
    answer.toSet == found.toSet
  }

  def evalPerformance(solver :EASolver) : Double = {
    val results : List[List[String]] = testCases map solver.solve
    val total  = results.length

    val suc :Int = (testCases zip results) count isSuccess
    (suc.toDouble/total)
  }

  // return TP, FP, FN
  def countPR(arg: (EACase, List[String])) : (Int,Int,Int) = {
    val testcase = arg._1
    val candidate : Set[Int] = tool.allEntity(testcase.targetSent::testcase.context) map entityDict.getGroup
    val answer : Set[Int] = (arg._1.entity map entityDict.getGroup).toSet
    val found : Set[Int] = (arg._2 map entityDict.getGroup).toSet
    val explicit_entity : Set[Int] = (entityDict.extractFrom(testcase.targetSent) map entityDict.getGroup).toSet
    val real_hidden = answer &~ explicit_entity
    val found_hidden = found &~ explicit_entity

    val tp: Set[Int] = real_hidden & found_hidden
    val fp = found_hidden &~ real_hidden
    val fn = real_hidden &~ found_hidden
    (tp.size, fp.size, fn.size)
  }

  def getRecallPrecision(solver:EASolver) : (Double,Double) = {
    val results : List[List[String]] = testCases map solver.solve
    val counts = ((testCases zip results) map countPR).unzip3
    val (tp, fp, fn) : (Int,Int,Int) = (counts._1.sum, counts._2.sum, counts._3.sum)
    val precision : Double = tp.toDouble / (tp+fp)
    val recall : Double = tp.toDouble / (tp+fn)
    (precision, recall)
  }

  def showResult(solver : EASolver) = {
    val results : List[List[String]] = testCases map solver.solve
    def show(item : (EACase,List[String])) : Unit= {

      val answer :String = item._1.entity.mkString(",")
      val strResult :String = item._2 match {
        case Nil => "None"
        case list => list.mkString(",")
      }

      val sentence = item._1.targetSent
      if(isSuccess(item))
        println(s"$sentence : $strResult (Correct)")
      else
        println(s"$sentence : Result=[$strResult] , but answer = [$answer]")
    }
    (testCases zip results) foreach show
  }
}
