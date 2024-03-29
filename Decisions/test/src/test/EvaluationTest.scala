package decisions
import utest._

import probability_monad._
import smile.classification._
import smile.math.MathEx.{logistic, min}
import scala.util

import com.github.sanity.pav.PairAdjacentViolators._
import com.github.sanity.pav._

import java.io._
import decisions._, EvalUtils._, TransactionsData._, AUC._, To._, Concordance._
import decisions.Shared._, LinAlg._, Stats._, FileIO._, RowLinAlg._, MatLinAlg._, CollectionsStats._

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

object helper{
    def sample(data: Row, perc: Double) = {
        require(0 < perc && perc < 1)
        val mask = Distribution.bernoulli(perc).sample(data.size)
        data zip(mask) filter{case (v,m) => m == 1} map(_._1)
    }

    object ECE{
        val pDevT = Array(0.9, 0.98, 0.85, 0.89, 0.65, 0.64, 0.32, 0.38, 0.30)
        val yDevT = Array(1, 1, 0, 1, 1, 0, 0, 0, 1)
        val expectedFrequency = Seq((0.32 + 0.38 + 0.30)/3.0, (0.65 + 0.64)/2.0 + (0.85 + 0.89)/2.0 + (0.98 + 0.90)/2.0)
        val expectedAccuracy = Seq((0 + 0 + 1)/3.0, (0 + 1)/2.0 + (1 + 0)/2.0 + (1 + 1)/2.0)
    }


    def gaussianScores: Tuple2[Vector[Double], Vector[Int]] = {
        object RepeatableDistribution extends Distributions(new scala.util.Random(54321))

        val targets = for {
            logOdds <- RepeatableDistribution.normal
            theta_1 = 2.0 + 2.0*logOdds
            } yield theta_1

        val nonTargets = for {
            logOdds <- RepeatableDistribution.normal
            theta_2 = -2.0 + 2.0*logOdds
            } yield theta_2

        val scores = (targets.sample(2000) ++ nonTargets.sample(10000)).toVector
        val pos: IndexedSeq[Int]  = for (i <- 1 to 2000) yield 1 
        val neg: IndexedSeq[Int]  = for (i <- 1 to 10000) yield 0
        val labels = pos.toVector ++ neg.toVector

        (scores, labels)
    }
    // Values from PYLLR after runnning the scores likelihoods generated above
    val steppyBer = Vector(0.07743527, 0.10648799, 0.13123004, 0.15037729, 0.16095   , 0.15255076, 0.1316486 , 0.10093192, 0.07696708)
    val steppyMajorityErrorRate = Vector(0.1192 , 0.18243, 0.26894, 0.37754, 0.5    , 0.37754, 0.26894, 0.18243, 0.1192)
    val convexHullBER = Vector(0.07697106, 0.10429837, 0.1300508 , 0.14831377, 0.15985   , 0.15136779, 0.13088949, 0.10054549, 0.07478175)
    val eer = 0.1603176277530788


    object PAVTestData {
        val x =  Vector(0.02, 0.1, 0.18, 0.2, 0.27, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)
        val y = Vector(0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
        val mean =  Vector(0.0, 1/3.0, 1/3.0, 1/3.0, 1/2.0, 1/2.0, 2/3.0, 2/3.0, 2/3.0, 3/4.0, 3/4.0, 3/4.0, 3/4.0, 1.0, 1.0)
    }

    object ROC{
        val w0Cnt = Vector(2,10,15,0,8,6,0,0,0) map(_.toDouble)
        val w1Cnt = Vector(0,0,0,3,0,4,8,5,2) map(_.toDouble)
        val thresh = Vector(-3,-2,-1,0,1,2,3,4,5)

        /* p(s>cutOff | w_i)

           Calculate tpr and fpr using the same operations as in
           the Tradeoff class methods but in a different order.

           Unit test order is <Reverse then accumulate then pro-rate>
           Tradeoff order  is <pro-rate then accumulate then one-minus then reverse>
        */
        def rhsRate(cnt: Row): Row = {
            val N = cnt.sum
            cnt
            .reverse
            .scanLeft(0.0)(_+_)
            .map(_ / N.toDouble)
        }
        val fpr = rhsRate(w0Cnt)
        val tpr = rhsRate(w1Cnt)
        val expected = Matrix(fpr,tpr)
    }

    object MinimiseRisk{
        def splitScores(data: List[Score]): Tuple2[Row,Row] = {
            val tarS = data.filter(_.label==1).map(_.s).toVector
            val nonS = data.filter(_.label==0).map(_.s).toVector
            (nonS,tarS)            
        }      

        val pa = AppParameters(0.5,1,1)
        def hiAUCdata: Distribution[List[Score]] = HighAUC.normalLLR.repeat(50)

        val nbins = 10
        
        def score2minrisk(data: List[Score]): Tuple2[Double,Double] = {
            val tarS = data.filter(_.label==1).map(_.s).toVector
            val nonS = data.filter(_.label==0).map(_.s).toVector
            
            val min = (tarS ++ nonS).min
            val max = (tarS ++ nonS).max
            val w0Cnts = histogram(nonS,nbins,min,max).map(_._2).toVector
            val w1Cnts = histogram(tarS,nbins,min,max).map(_._2).toVector
            val histThresh = histogram(tarS,nbins,min,max).map(_._1.toDouble).toVector            
            val to = Tradeoff(w1Cnts, w0Cnts,histThresh)
            val actual = to.minRisk(pa)

            val PP = Matrix(Row(pa.p_w1, 1-pa.p_w1))
            val expected = minRiskBruteForce(PP,to.asPmissPfa,pa.Cfa,pa.Cmiss)

            (actual,expected)
        }

        val simulate = for (dataset <- hiAUCdata) yield score2minrisk(dataset)
    }
}

object EvaluationsTests extends TestSuite with TestHelper{ // add EvalUtils as Trait
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
                val expected = helper.convexHullBER
                assert(expected.zip(result).filter{tup => tup._1 ~= tup._2}.size == result.size)
            }
            test("EER"){
                implicit val precision = Precision(0.01)
                val result = convexHull.EER
                val expected = helper.eer
                assert(expected ~= result)
            }
        }
        test("minimizeScalar") {
            // Based on the scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
            val f: (Double => Double) = x => (x - 2) * x * (x + 2)*(x + 2)
            val optimized = new BrentOptimizerScalarWrapper(f, -5, 5, minimize=true)
            val pointExpected = 1.2807764040333458
            val valueExpected = -9.914949590828147

            implicit val precision = Precision(0.001)
            assert( optimized.optimumPoint ~= pointExpected)
            assert( optimized.optimumValue ~= valueExpected)
        }
        test("PAVwrapper"){
            // The test case data is based on "PAV and the ROC convex hull" by Fawcett & Niculescu
            val pavFit = new PairAdjacentViolatorsWrapper(helper.PAVTestData.x, helper.PAVTestData.y)
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

        test("ROC"){
            val expected = helper.ROC.expected
            val to = Tradeoff(helper.ROC.w1Cnt,helper.ROC.w0Cnt,helper.ROC.thresh)
            val actual = to.asROC
            assert(expected(0).zip(actual(0)).filter{tup => tup._1 ~= tup._2}.size == actual(0).size)
            assert(expected(1).zip(actual(1)).filter{tup => tup._1 ~= tup._2}.size == actual(0).size)
        }

        test("MinimiseRisk"){
            val numSims = 10
            val minRisk = helper.MinimiseRisk.simulate
            val testCases = minRisk.sample(numSims)
            testCases.foreach{case (actual,expected) => actual ~= expected}
        }

        test("ConcordanceProbability"){
            val data = helper.MinimiseRisk.hiAUCdata.sample(1)(0)
            val (non, tar) = helper.MinimiseRisk.splitScores(data)

            assert(naiveA(non,tar) == smartA(non,tar))
        }

        test("CalibrationError"){
            val metrics = binnedAccuracy(helper.ECE.pDevT, helper.ECE.yDevT, EvalUtils.binBy0_10)
            val resultAccuracy = metrics.map(_.accuracy)
            val resultFrequency = metrics.map(_.frequency)
            val expectedAccuracy = helper.ECE.expectedAccuracy
            val expectedFrequency = helper.ECE.expectedFrequency
            assert(expectedAccuracy.zip(resultAccuracy).filter{tup => tup._1 ~= tup._2}.size == resultAccuracy.size)
            assert(expectedFrequency.zip(resultFrequency).filter{tup => tup._1 ~= tup._2}.size == resultFrequency.size)
        }
    }
}