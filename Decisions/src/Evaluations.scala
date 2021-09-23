package decisions

import collection.JavaConverters._
import scala.language.postfixOps
import scala.util

import plotly._, element._, layout._, Plotly._ 
import be.cylab.java.roc.Roc
import com.github.sanity.pav.PairAdjacentViolators._
import com.github.sanity.pav._
import smile.classification._
import smile.math.MathEx.{logistic, min}
import scala.math.{abs, floor, round, log}

import decisions.TransactionsData._
import decisions.Shared._, LinAlg._, Stats._, FileIO._, RowLinAlg._, MatLinAlg._, CollectionsStats._

import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.MaxEval 
import org.apache.commons.math3.optim.univariate.SearchInterval
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction
import org.apache.commons.math3.optim.univariate.BrentOptimizer 

case class AppParameters(p_w1: Double, Cmiss: Double, Cfa: Double)

/*
Every error estimator needs a logodds to probability converter
Optimal errors need convex hull coordinates to make calculation more efficient
Observed errors use the steppy curve to compute the observed bayes error rates

The curve is the latent representation; a representation of the classifier in the 
(pFa, pMiss space). The observed errors don't explicitly compute the operating points
from the scores but do compute them from the prior log-odds (equivalent??)
*/
trait ErrorEstimator{
    def logoddsToProbability(priorLogOdds: Row): Matrix = {
        val pTar: Row = priorLogOdds.map(logistic)
        val pNon: Row = priorLogOdds.map(x => logistic(-x))
        Matrix(pTar, pNon)
    }
    def bayesErrorRate: Row
}

class SteppyCurve(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double]) extends ErrorEstimator {
        import RowLinAlg._, MatLinAlg._

        val PP: Matrix = logoddsToProbability(priorLogOdds)
        val pMisspFa = EvalUtils.pMisspFaPoints(scores, labels, priorLogOdds)
        def bayesErrorRate = PP.transpose * pMisspFa.transpose map{case e: Row => e reduceLeft(_+_)}
        def majorityErrorRate = PP.transpose.map(row => row.min)
}

trait ConvexHull extends ErrorEstimator {
    def convexHullCoordinates: Vector[Vector[Double]]
}

class PAV(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double]) extends ConvexHull {
    import RowLinAlg._, MatLinAlg._

    private def countTargets: Vector[Int] =
        pavFit.bins.map{case po: Point => po.getWeight.toDouble * po.getY toInt}
            .toVector

    private def countNonTargets: Vector[Int] =
        targets.zip(pavFit.bins).map{case (count,po) =>  (po.getWeight - count).toInt}
              .toVector

    val pavFit = new EvalUtils.PairAdjacentViolatorsWrapper(scores, labels)
    val nbins = pavFit.bins.size
    val targets = countTargets
    val nonTars = countNonTargets
    val T = targets.reduce(_ + _)
    val N = nonTars.reduce(_ + _)

    def convexHullCoordinates: Vector[Vector[Double]] = {
        val pmiss = targets.scanLeft(0)(_ + _).map{_ / T.toDouble}
        val pfa = nonTars.scanLeft(0)(_ + _).map{1 - _ / N.toDouble}
        Vector(pmiss, pfa)
    }

    val pMisspFa: Matrix = convexHullCoordinates

    /*
    *  PP is (2,m)
    */   
    def bayesErrorRate: Row = {
        val PP = logoddsToProbability(priorLogOdds)
        val E = PP.transpose at pMisspFa
        val ber = for (err <- E) yield err.min
        ber
    }
    def EER: Double = {
        val objectiveFunction: (Double => Double) = x => {
            val PP: Matrix = logoddsToProbability(Vector(x))
            val E = PP.transpose at pMisspFa
            val minn: Row = for (err <- E) yield err.min
            minn(0)
        }
        val maximised = new EvalUtils.BrentOptimizerScalarWrapper(objectiveFunction, priorLogOdds(0), priorLogOdds.last, minimize=false)
        maximised.optimumValue
    }

    def scoreVSlogodds: Tuple2[Row, Row] = pavFit.bins.map{case po => (po.getX, logit(po.getY))}.unzip
}


object Tradeoff{

    import EvalUtils._
    import CollectionsStats._

    case class Point(x: Double,y: Double)
    case class Segment(p1: Point, p2: Point)

    case class Tradeoff(w1Counts: Vector[Double], w0Counts: Vector[Double], thresholds: Vector[Double]) {
        val N = w1Counts.size
        require(w0Counts.size == N)
        require(thresholds.size == N)

        /* Various ways to present the score distributions */

        val asCCD: Matrix = {
            val w0pdf = pdf(w0Counts)
            val w1pdf = pdf(w1Counts)
            Vector(w0pdf,w1pdf)
        }

        val asLLR: Row = {
            val infLLR = logodds((pdf(w0Counts),pdf(w1Counts)))
            clipToFinite(infLLR)
        }

        val asPmissPfa: Matrix = {
            val pMiss = cdf(w1Counts) // lhsArea of p(x|w1)
            val pFa = rhsArea(w0Counts) // Same as fpr but in the asc order of scores
            Vector(pMiss,pFa)
        }

        val asROC: Matrix = {
            val fpr = (rhsArea andThen decreasing)(w0Counts)
            val tpr = (rhsArea andThen decreasing)(w1Counts)
            Vector(fpr,tpr)
        }

        def asDET: Matrix = ???
        
        /* Given application parameters, return optimal threshold and the corresponding expected risk */

        /* Find score cut-off point that minimises Risk by finding
        where LLRs intersect -θ.
        Return the corresponding index in the score Vector.
        */
        def minusθ(pa: AppParameters) = -1*paramToTheta(pa)

        def argminRisk(pa: AppParameters): Int = this.asLLR.getClosestIndex(minusθ(pa))

        def expectedRisks(pa: AppParameters): Row = {
            val risk: Tuple2[Double,Double] => Double = paramToRisk(pa)
            this.asPmissPfa.transpose.map{case Vector(pMiss,pFa) => risk((pMiss,pFa))}
        }

        def minS(pa: AppParameters): Double = {
            val ii = argminRisk(pa)
            thresholds(ii)
        }

        def minRisk(pa: AppParameters): Double = {
            val ii = argminRisk(pa)
            val bestPmissPfa = (this.asPmissPfa.apply(0)(ii),this.asPmissPfa.apply(1)(ii))
            paramToRisk(pa)(bestPmissPfa)
            //== expectedRisks(pa)(ii)
        }

        def ber(p_w1: Double): Double = minRisk(AppParameters(p_w1,1,1))

        def affine(y1: Double, x1: Double, x2: Double, slope: Double) = y1 + (x2-x1)*slope

        def isocost(pa: AppParameters): Segment = {
            val slope = exp(-1*paramToTheta(pa))
            val ii = argminRisk(pa)
            val roc = this.asROC
            val (bfpr, btpr) = (roc(0).reverse(ii), roc(1).reverse(ii)) // .reverse because roc curves are score inverted
            val (x1,x2) = (roc(0)(0), roc(0).last)
            val (y1, y2) = (affine(btpr,bfpr,x1,slope),
                            affine(btpr,bfpr,x2,slope)
            )
            Segment(Point(x1,y1),Point(x2,y2))
        }
    }    
}

object EvalUtils extends{
    import RowLinAlg._, MatLinAlg._
    /*  - Method to get the PAV bins, i.e. the PAV-merged points.
        - Handle kotlin to scala type conversion to hide it from the main PAV class.
    */
    class PairAdjacentViolatorsWrapper(val scores: Vector[Double], val labels: Vector[Int]) {
        def vecToPoints(x: Vector[Double], y: Vector[Int]): Vector[Point] =
            x.zip(y).map{case (x,y) => new Point(x,y)}
        
        def getBins = pav.getIsotonicPoints.asScala.toVector
        
        val inputPoints = vecToPoints(scores,labels).toIterable.asJava
        val pav = new PairAdjacentViolators(inputPoints)
        val bins: Vector[Point] = getBins
    }

    class BrentOptimizerScalarWrapper(
        val objectiveFunction: (Double => Double), 
        val intervalMin: Double, 
        val intervalMax: Double, 
        val iterations: Int = 500,
        val minimize: Boolean = true
        ) {
        val objective = if(minimize == true){
            new UnivariateObjectiveFunction(x => -1*objectiveFunction(x))
        } else {
            new UnivariateObjectiveFunction(x => objectiveFunction(x))
        }            
        val goal = GoalType.MAXIMIZE
        val maxEval = new MaxEval(iterations)
        val optimizer = new BrentOptimizer(0.001, 0.001)
        val interval = new SearchInterval(intervalMin, intervalMax, 0.5*(intervalMax - intervalMin))
        val optimized = optimizer.optimize(objective, goal, interval, maxEval)
        def optimumPoint: Double = optimized.getPoint
        def optimumValue: Double = if(minimize == true){
            objectiveFunction(optimumPoint)
        } else {
            optimized.getValue
        }   
    }

    case class Reliability(bin: Double, frequency: Double, accuracy: Double, count: Integer)
    val avg: (Array[Double] => Double) = values => values.sum/values.size.toDouble
    val binTheValue: (Double => (Double => Double)) = width => (i => round(i / width).toInt*width)
    val binBy0_05: (Double => Double) = binTheValue(0.05)
    val binBy0_10: (Double => Double) = binTheValue(0.10)

    def binnedAccuracy(pDev: Array[Double], yDev: Array[Int], binValue: Double => Double): Seq[Reliability] = {
        val sorted: Array[(Double, Int)] = pDev.zip(yDev).sorted
        val binned = sorted.map{case (proba, label) => (binValue(proba), proba, label)}

        val bins = binned.groupMap(_._1)(_._3).map{case (bin, label) => bin}
        val counts = binned.groupMap(_._1)(_._3).map{case (bin, label) => label.size}
        val accuracy = binned.groupMap(_._1)(_._3).map{case (bin, label) => avg(label.map(_.toDouble))}
        val frequency = binned.groupMap(_._1)(_._2).map{case (bin, proba) => avg(proba)}

        // Combine the computations above.
        val reliability = bins.zip(frequency.zip(accuracy.zip(counts))).map{
            case (bin, (freq, (acc, cnt))) => Reliability(bin, freq, acc, cnt)
        }

        reliability.toSeq.sortBy(_.bin) // Too much sorting going on
    }


    /* Expected Calibration Error
        ∑_{bin1}^{binB}{ num_bins/N * abs(ac    c(b)−freq(b)) }
    */
    def ECE(data: Seq[Reliability]): Double = {
        val N = data.map(_.count.toDouble).sum
        val ece = data.map{bin => (abs(bin.accuracy - bin.frequency)) * (bin.count / N) }
                        .reduce( _ + _)
        ece
    }     

    def ordinalRank(arr: Seq[Double]): Seq[Int] = {
        val withIndices = arr.zipWithIndex
        val valueSorted = withIndices.sortBy(_._1)
        val withRank = valueSorted.zipWithIndex
        val indexSorted = withRank.sortBy(_._1._2)
        for((valueAndIndexTuple, rank) <- indexSorted) yield rank + 1
    }

    def pMisspFaPoints(scores: Row, labels: Vector[Int], priorLogOdds: Row): Matrix = {
        val thr = priorLogOdds.map(-1 * _)
        // TODO: data class with positive and negative types
        val tar = scores.zip(labels).filter(_._2==1).map(_._1)
        val non = scores.zip(labels).filter(_._2==0).map(_._1)

        val D = priorLogOdds.size
        val T = tar.size
        val N = non.size
        val DdownTo1 = (D to 1 by -1)

        val rk = ordinalRank((thr ++ tar).map(_.toDouble))
        val rkD = rk.take(D)
        val pMiss = for ((r,v) <- rkD zip DdownTo1) yield (r-v.toFloat) / T

        val rk2 = ordinalRank((thr ++ non).map(_.toDouble))
        val rkD2 = rk2.take(D)
        val pFa = for ((r,v) <- rkD2 zip DdownTo1) yield (N - r.toFloat + v) / N
        //(pMiss.map(_.toDouble).toVector, pFa.map(_.toDouble).toVector)
        Vector(pMiss.toVector, pFa.toVector)
    }

    /* Minimum expected risks for a Sequence of prior probabilities

        Brute force method i.e. compute the expected risk at every operating point
        then look for the minimum.
        Should give the same result as the likelihood approach llr = -θ.
        In practice it's ok to use this approach, especially with PAV bins, 
        because there are not many points to traverse.
        
        * PP is (n,2) i.e. rows of (p_w0,p_w1)
        * pMissPfa is (2,m) i.e. columns of ((Pmiss),(Pfa))
        * Returns (PP * Matrix(Row(Cfa,Cmiss)) ) @ pMisspFa
    */
    def minRiskBruteForce(PP: Matrix, pMissPfa: Matrix, Cfa: Double, Cmiss: Double): Double = {
        val priorCosts: Matrix = for (row <- PP) yield row match {
            case Vector(p_w1,p_w0) => Row(p_w1*Cmiss,p_w0*Cfa)
            case _ => Row(-10,-10) // Something's off, insert crazy values to flag the issue
        }
        priorCosts at pMissPfa apply(0) min
    }

    def cost(p: AppParameters, actual: User, pred: User): Double = pred match {
            case Fraudster if actual == Regular => p.Cfa
            case Regular if actual == Fraudster => p.Cmiss
            case _ => 0.0
    }

    def paramToTheta(p: AppParameters): Double = log(p.p_w1/(1-p.p_w1)*(p.Cmiss/p.Cfa))

    /*  Expected Risk using evaluation data at a particular operating point

        E(r) = Cmiss.p(ω1).∫p(x<c|ω1)dx + Cfa.p(ω0).∫p(x>c|ω0)dx
            = Cmiss.p(ω1).Pmiss + Cfa.p(ω0).Pfa
        c: cutoff point to evaluate

        @param pa the application's prior probability and cost of miss and false alarm
        @param operatingPoint the probability of miss and false alarm
        @return the expected risk
    */        
    def paramToRisk(pa: AppParameters)(operatingPoint: Tuple2[Double,Double]): Double = 
            pa.p_w1*operatingPoint._1*pa.Cmiss + (1-pa.p_w1)*operatingPoint._2*pa.Cfa    
}
