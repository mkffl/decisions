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


object To{
    import EvalUtils._
    case class Point(x: Double,y: Double)
    case class Segment(p1: Point, p2: Point)

    /*
        w0Counts: counts of non-target labels
        w1Counts: counts of target labels
        thresh: bins
    */
    def makeHisTo(loEval: Row, yEval: Vector[Int], numBins: Int = 30): Tradeoff = {
        val tarPreds = loEval zip yEval filter{case (lo,y) => y == 1} map {_._1} 
        val nonPreds = loEval zip yEval filter{case (lo,y) => y == 0} map {_._1}            

        val min=loEval.min
        val max=loEval.max
        val w0Counts = histogram(nonPreds,numBins,min,max).map(_._2).toVector
        val w1Counts = histogram(tarPreds,numBins,min,max).map(_._2).toVector
        val thresh = histogram(tarPreds,numBins,min,max).map(_._1.toDouble).toVector
        
        Tradeoff(w1Counts,w0Counts,thresh)
    }

    case class Tradeoff(w1Counts: Row, w0Counts: Row, thresholds: Row) {
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
        
        /* Optimal threshold given application parameters.

            Find score cut-off point that minimises Risk by finding
            where LLRs intersect -θ. Takes the closest LLR value rather 
            than the first value above or equal to -θ.

            Note that values based on cdf, e.g. PmissPfa or ROC, need to
            use (argminRisk+1) because they have one more value.
        
            @Return The position in the 0-indexed score vector.
        */
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
            val bestPmissPfa = (this.asPmissPfa.apply(0)(ii+1),this.asPmissPfa.apply(1)(ii+1))
            paramToRisk(pa)(bestPmissPfa)
        }

        def ber(p_w1: Double): Double = minRisk(AppParameters(p_w1,1,1))

        def isocost(pa: AppParameters): Segment = {
            val slope = exp(minusθ(pa))
            val ii = argminRisk(pa)
            val roc = this.asROC
            val (bfpr, btpr) = (roc(0).reverse(ii+1), roc(1).reverse(ii+1)) // .reverse because roc curves are score inverted
            val (x1,x2) = (roc(0)(0), roc(0).last)
            val (y1, y2) = (affine(btpr,bfpr,x1,slope),
                            affine(btpr,bfpr,x2,slope)
            )
            Segment(Point(x1,y1),Point(x2,y2))
        }
    }    
}

object Concordance{
    // Common Language Effect size
    // Naive and wmw-based computations
    def getPermutations(A: Row, B: Row): Vector[Tuple2[Double,Double]] = for {
            a <- A
            b <- B
        } yield (a,b)

    /* count [score_w1 > score_w0] */
    def TarSupN(non:Row, tar:Row): Int = getPermutations(non,tar) filter {score => score._2 > score._1} size
    
    /* Estimate P(score_w1 > score_w0) */
    def naiveA(non: Row, tar: Row): Double = {
        val num = TarSupN(non,tar)
        val den = non.size*tar.size
        num/den.toDouble
    }

    /* Rank values with tied averages
        Inpupt: Vector(4.5, -3.2, 1.2, 5.6, 1.2, 1.2)
        Output: Vector(5,   -1,   3,   6,   3,   3  )
    */    
    def rankAvgTies(input: Row): Row = {
        // Helper to identify fields in the tuple cobweb
        case class Rank(value: Double,index: Integer,rank: Integer)

        val enhanced = input.zipWithIndex.
                        sortBy(_._1).zipWithIndex.
                        map{case ((lo,index),rank) => Rank(lo,index,rank+1)}
        val avgTies = enhanced.groupBy(_.value).
                    map{ case (value, v) => (value, v.map(_.rank.toDouble).sum / v.map(_.rank).size.toDouble)}
        val joined = for {
            e <- enhanced
            a <- avgTies
            if (e.value == a._1)
        } yield (e.index,a._2)

        joined.sortBy(_._1).map(_._2.toInt)
    }    

    /* Wilcoxon Statistic, also named U */
    def wmwStat(s0: Row, s1: Row): Int = {
        val NTar = s1.size
        val ranks = rankAvgTies(s0 ++ s1)
        val RSum = ranks.takeRight(NTar).sum
        val U = RSum - NTar*(NTar+1)/2
        U toInt
    }
    
    /* Estimate P(score_w1 > score_w0) */
    def smartA(non:Row, tar:Row) = {
        val den = non.size*tar.size
        val U = wmwStat(non,tar)
        val A = U.toDouble/den
        A
    }

}

object EvalUtils{
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

    /* Detection Cost Function (DCF)

    @param pa the application's prior probability and cost of miss and false alarm
    @param actual the ground truth
    @param pred the predicted value
    */
    def cost(p: AppParameters, actual: User, pred: User): Double = pred match {
            case Fraudster if actual == Regular => p.Cfa
            case Regular if actual == Fraudster => p.Cmiss
            case _ => 0.0 //TODO: remove that case
    }

    /*
        θ = log{p_w1*Cmiss/p_w0*Cfa} = -log{δ}
    */    
    def paramToθ(pa: AppParameters): Double = log(pa.p_w1/(1-pa.p_w1)*(pa.Cmiss/pa.Cfa))

    /*
        RR = p_w0*Cfa/p_w1*Cmiss
    */
    def paramToRiskRatio(p: AppParameters): Double = ((1-p.p_w1)*p.Cfa)/(p.p_w1*p.Cmiss)

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

    def minusθ(pa: AppParameters) = -1*paramToθ(pa)

    def affine(y1: Double, x1: Double, x2: Double, slope: Double) = y1 + (x2-x1)*slope

    /* Isocost for a majority classifier.
        Given application parameters, return two points on the equal-risk line
        that crosses (d,d) in the ROC space.

        The underlying equation is (tpr-d) = (fpr-d)*rr
        with rr = p_w0*Cfa / p_w1*Cmiss.

        If the all_w0 classifier has a lower risk, which happens if rr >= 1
        then the line goes thru (0,0) and the other point is on (a,1), with a
        determined in code below.

        if the all_w1 has lower risk then its line goes thru (0,a) and (1,1).
        
        @Returns a Segment object with Point 1 and Point 2 described above.
    */
    def majorityIsocost(pa: AppParameters): To.Segment = {
        val rr = paramToRiskRatio(pa)
        if (rr >= 1){
            val p1 = To.Point(0,0)
            val p2 = To.Point(1/rr,1)
            To.Segment(p1,p2)
        }
        else {
            val p1 = To.Point(0,1-rr)
            val p2 = To.Point(1,1)
            To.Segment(p1,p2)
        }
    }
}
