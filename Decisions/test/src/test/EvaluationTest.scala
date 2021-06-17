package decisions
import utest._

import probability_monad._
import smile.classification._
import smile.math.MathEx.{logistic, min}
import scala.util

import decisions._
import java.io._

trait TestHelper {    
    case class Precision(val p:Double)
    class withAlmostEquals(d:Double) {
        def ~=(d2:Double)(implicit p:Precision) = (d-d2).abs <= p.p
    }
    implicit def add_~=(d:Double) = new withAlmostEquals(d)
    implicit val precision = Precision(0.0001)
}

object helper {
    def gaussianScores: Tuple2[Vector[Double], Vector[Int]] = {
        object RepeatableDistribution extends Distributions(new scala.util.Random(54321))

        val targets = for {
            logOdds <- RepeatableDistribution.normal
            score = logistic(2.0 + logOdds)
            } yield score

        val nonTargets = for {
            logOdds <- RepeatableDistribution.normal
            score = logistic(-2.0 + logOdds)
            } yield score

        val scores = (targets.sample(100) ++ nonTargets.sample(10000)).toVector
        val pos: IndexedSeq[Int]  = for (i <- 1 to 100) yield 1 
        val neg: IndexedSeq[Int]  = for (i <- 1 to 10000) yield 0
        val labels = pos.toVector ++ neg.toVector

        (scores, labels)
    }

    val steppyBer = Vector(0.11920292, 0.18242552, 0.26894142, 0.01937754, 0.5, 0.37754067, 0.26894142, 0.18242552, 0.11920292)
    val steppyMajorityErrorRate = Vector(0.11920292, 0.18242552, 0.26894142, 0.37754067, 0.5, 0.37754067, 0.26894142, 0.18242552, 0.11920292)
}

object EvaluationsTests extends TestSuite with TestHelper{
  def tests = Tests {
    val (scores, labels) = helper.gaussianScores
    val ploRange: Vector[Double] = (BigDecimal(-2.0) to BigDecimal(2.0) by 0.5).map(_.toDouble).toVector
    val steppy = new SteppyCurve(scores, labels, ploRange)
        test("steppyBER") {
            val expected = helper.steppyBer
            val result = steppy.bayesErrorRate
            assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size) //TODO: add a shouldBeApprox method
        }
        test("steppyMajority"){
            val expected = helper.steppyMajorityErrorRate
            val result = steppy.majorityErrorRate
            assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size) //TODO: add a shouldBeApprox method
        }
    }
}


