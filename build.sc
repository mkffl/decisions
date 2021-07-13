// build.sc
import mill._, scalalib._
import coursier.maven.MavenRepository

object Decisions extends ScalaModule{
  def scalaVersion = "2.13.5"

  def repositories = super.repositories ++ Seq(
    MavenRepository("https://jitpack.io")
  )

  def ivyDeps = Agg(
      ivy"com.typesafe:config:1.4.0",
      ivy"be.cylab:java-roc:0.0.6",
      ivy"org.jliszka:probability-monad_2.13:1.0.4",
      ivy"com.github.haifengl:smile-core:2.6.0",
      ivy"com.github.haifengl:smile-plot:2.6.0",
      ivy"com.github.haifengl:smile-io:2.6.0",
      ivy"org.plotly-scala::plotly-render:0.8.1",
      ivy"com.github.sanity:pairAdjacentViolators:1.4.16",
      ivy"org.apache.commons:commons-math3:3.6.1",
  )

  object test extends Tests{
    def ivyDeps = Agg(ivy"com.lihaoyi::utest:0.7.4")
    def testFrameworks = Seq("utest.runner.Framework")
  }
}
