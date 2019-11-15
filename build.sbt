import scalapb.compiler.Version.scalapbVersion

name := "spark-ICA"

version := "0.2"

scalaVersion := "2.11.12"

PB.targets in Compile := Seq(
  scalapb.gen() -> (sourceManaged in Compile).value
)

val sparkVersion = "2.4.0"
// https://mvnrepository.com/artifact/org.apache.spark/spark-core

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x                             => MergeStrategy.first
}

libraryDependencies ++= Seq(
  "org.apache.spark"     %% "spark-sql"                 % sparkVersion % "provided",
  "org.apache.spark"     %% "spark-core"                % sparkVersion % "provided",
  "org.apache.spark"     %% "spark-mllib"               % sparkVersion % "provided",
  "org.scalatest"        %% "scalatest"                 % "3.0.5" % "test",
  "com.thesamet.scalapb" %% "scalapb-runtime-grpc"      % scalapbVersion,
  "com.thesamet.scalapb" %% "scalapb-runtime"           % scalapbVersion % "protobuf",
  "com.jcraft"           % "jsch"                       % "0.1.53"
)

libraryDependencies += "org.scalameta" %% "scalafmt-dynamic" % "2.2.1"
