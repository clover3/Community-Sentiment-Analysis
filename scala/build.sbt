name := "scala"


resolvers ++= Seq(
  "snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  "releases" at "http://oss.sonatype.org/content/repositories/releases",
  "maven" at "https://repo1.maven.org/maven2"
)

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % "2.12.1",
  "org.scala-lang.modules" % "scala-xml_2.12" % "1.0.6"
)
libraryDependencies += "com.twitter.penguin" % "korean-text" % "4.4.2"
libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.3.4"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
libraryDependencies += "org.scala-lang" % "scala-library" % "2.11.7"


version := "1.0"

scalaVersion := "2.12.1"


