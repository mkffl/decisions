package decisions
import utest._

import probability_monad._
import smile.classification._
import smile.math.MathEx.{logistic, min}
import scala.util

import com.github.sanity.pav.PairAdjacentViolators._
import com.github.sanity.pav._

import decisions._
import java.io._

trait TestHelper {    
    case class Precision(val p:Double)
    class withAlmostEquals(d:Double) {
        def ~=(d2:Double)(implicit p:Precision) = (d-d2).abs <= p.p
    }
    implicit def add_~=(d:Double) = new withAlmostEquals(d)
    implicit val precision = Precision(0.0001)

    def almostEqual(d1: Double, d2: Double, threshold: Double = 0.01): Boolean =
        (d1-d2).abs <= threshold
}

object helper {
    def gaussianScores: Tuple2[Vector[Double], Vector[Int]] = {
        object RepeatableDistribution extends Distributions(new scala.util.Random(54321))

        val targets = for {
            logOdds <- RepeatableDistribution.normal
            theta_1 = 2.0 + 2.0*s
            } yield score

        val nonTargets = for {
            logOdds <- RepeatableDistribution.normal
            theta_2 = -2.0 + 2.0*s
            } yield score

        val scores = (targets.sample(100) ++ nonTargets.sample(10000)).toVector
        val pos: IndexedSeq[Int]  = for (i <- 1 to 100) yield 1 
        val neg: IndexedSeq[Int]  = for (i <- 1 to 10000) yield 0
        val labels = pos.toVector ++ neg.toVector

        (scores, labels)
    }
    // Values from PYLLR after runnning the scores likelihoods generated above
    val steppyBer = Vector(0.07964, 0.10648, 0.12938, 0.15323, 0.16108, 0.15174, 0.13339, 0.10423, 0.08058)//Vector(0.11920292, 0.18242552, 0.26894142, 0.01937754, 0.5, 0.37754067, 0.26894142, 0.18242552, 0.11920292)
    val steppyMajorityErrorRate = Vector(0.1192 , 0.18243, 0.26894, 0.37754, 0.5    , 0.37754, 0.26894, 0.18243, 0.1192)//Vector(0.11920292, 0.18242552, 0.26894142, 0.37754067, 0.5, 0.37754067, 0.26894142, 0.18242552, 0.11920292)
    val convexHullBER = Vector(0.07786, 0.10498, 0.12881, 0.14919, 0.15868, 0.14731, 0.128, 0.10397, 0.07802)//Vector(0.01169, 0.01416, 0.0169 , 0.01811, 0.01455, 0.01099, 0.00783, 0.00531, 0.00347)
    val eer = 0.1603176277530788


    object PAVTestData {
        val x =  Vector(0.02, 0.1, 0.18, 0.2, 0.27, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)
        val y = Vector(0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
        val mean =  Vector(0.0, 1/3.0, 1/3.0, 1/3.0, 1/2.0, 1/2.0, 2/3.0, 2/3.0, 2/3.0, 3/4.0, 3/4.0, 3/4.0, 3/4.0, 1.0, 1.0)
    }
}

object EvaluationsTests extends TestSuite with TestHelper{
    def tests = Tests {
        test("ErrorEstimators"){
            // Based on the ROCAnalysis Julia package: https://nbviewer.jupyter.org/github/davidavdav/ROCAnalysis.jl/blob/master/ROCAnalysis.ipynb
            val (scores, labels) = helper.gaussianScores
            val ploRange: Vector[Double] = (BigDecimal(-2.0) to BigDecimal(2.0) by 0.5).map(_.toDouble).toVector
            val steppy = new SteppyCurve(scores, labels, ploRange)
            val convexHull = new PAV(scores, labels, ploRange)

            test("observedDCF"){
                val expected = helper.steppyBer
                val result = steppy.bayesErrorRate
                assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size) //TODO: add a shouldBeApprox method
            }
            test("steppyMajority"){
                val expected = helper.steppyMajorityErrorRate
                val result = steppy.majorityErrorRate
                assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size) //TODO: add a shouldBeApprox method
            }
            test("minDCF"){
                implicit val precision = Precision(0.01)
                val result = convexHull.bayesErrorRate
                val expect = helper.convexHullBER
                assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size)
            }
            test("EER"){
                implicit val precision = Precision(0.01)
                val result = convexHull.EER
                val expect = helper.eer
                assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size)
            }
        }
        test("minimizeScalar") {
            // Based on the scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
            val f: (Double => Double) = x => (x - 2) * x * (x + 2)*(x + 2)
            val optimized = new utils.BrentOptimizerScalarWrapper(f, -5, 5, minimize=true)
            val pointExpected = 1.2807764040333458
            val valueExpected = -9.914949590828147

            implicit val precision = Precision(0.001)
            assert( optimized.optimumPoint ~= pointExpected)
            assert( optimized.optimumValue ~= valueExpected)
        }
        test("PAVwrapper"){
            // The test case data is based on "PAV and the ROC convex hull" by Fawcett & Niculescu
            val pavFit = new utils.PairAdjacentViolatorsWrapper(helper.PAVTestData.x, helper.PAVTestData.y)
            val result = for {
                bin <- pavFit.bins
                numValues = bin.getWeight.toInt
                predicted = bin.getY
                binValues = for (i <- 1 to numValues) yield predicted
                values <- binValues
            } yield values
            val expected = helper.PAVTestData.mean
            assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size)
        }
    }
}

