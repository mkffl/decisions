// build.sc
import mill._, scalalib._

object Decisions extends ScalaModule{
  def scalaVersion = "2.13.5"

  def ivyDeps = Agg(
      ivy"com.typesafe:config:1.4.0",
      ivy"be.cylab:java-roc:0.0.6",
      ivy"org.jliszka:probability-monad_2.13:1.0.4",
      ivy"org.jliszka:probability-monad_2.13:1.0.4",
      ivy"com.github.haifengl:smile-core:2.6.0",
      ivy"com.github.haifengl:smile-plot:2.6.0",
      ivy"org.plotly-scala::plotly-render:0.8.1",
  )

  object test extends Tests{
    def ivyDeps = Agg(ivy"com.lihaoyi::utest:0.7.4")
    def testFrameworks = Seq("utest.runner.Framework")
  }
}
