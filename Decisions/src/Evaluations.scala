package decisions
import plotly._, element._, layout._, Plotly._ 
import collection.JavaConverters._
import be.cylab.java.roc.Roc
import scala.util
import com.github.sanity.pav.PairAdjacentViolators._
import com.github.sanity.pav._
import smile.classification._
import smile.math.MathEx.{logistic, min}
import scala.math.{abs, floor, round}

import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.MaxEval 
import org.apache.commons.math3.optim.univariate.SearchInterval
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction
import org.apache.commons.math3.optim.univariate.BrentOptimizer 

object LinAlgebra {
    // TODO: find out how to use package objects 
    type Row = Vector[Double]
    type Matrix = Vector[Vector[Double]]

    def Row(xs: Int*) = Vector(xs: _*)
    def Matrix(xs: Row*) = Vector(xs: _*)

    implicit def intToDouble(mat: Vector[Vector[Int]]): Matrix = 
        for (row <- mat) 
            yield for (value <- row) 
                yield value.toDouble
    
    implicit def floatToDoubleRow(row: Vector[Float]): Row = 
        for (value <- row)
                yield value.toDouble

    /* Matrix product operation
    If matrix A is (m,n) and B is (n,p) then output is (m,p)
    Input matrices must be (m,n) and (n,p).
    */
    def matMul(A: Matrix, B: Matrix) = {
        for (row <- A) // (m,n)
        yield for(col <- B.transpose) // rotate to (p,n) to loop thru the cols
            yield row zip col map Function.tupled(_*_) reduceLeft (_+_)
    }

    def dotProduct(A: Matrix, B: Matrix) = {
        for ( (rowA, rowB) <- A zip B)
        yield rowA zip rowB map Function.tupled(_*_) reduceLeft (_+_)
    }
}



/*
Every error estimator needs a logodds to probability converter
Optimal errors need convex hull coordinates to make calculation more efficient
Observed errors use the steppy curve to compute the observed bayes error rates

The curve is the latent representation; a representation of the classifier in the 
(pFa, pMiss space). The observed errors don't explicitly compute the operating points
from the scores but do compute them from the prior log-odds (equivalent??)
*/
trait ErrorEstimator {
    import LinAlgebra._

    def logoddsToProbability(priorLogOdds: Vector[Double]): Matrix = {
        val pTar: Row = priorLogOdds.map(logistic)    
        val pNon: Row = priorLogOdds.map(x => logistic(-x))
        Vector(pTar, pNon)
    }
    def bayesErrorRate: Row
}

class SteppyCurve(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double]) extends ErrorEstimator {
        import LinAlgebra._

        val PP = logoddsToProbability(priorLogOdds).transpose
        val pMisspFa = EvalUtils.pMisspFaPoints(scores, labels, priorLogOdds).transpose
        def bayesErrorRate = dotProduct(PP, pMisspFa) //dot product or 'dotProductSum'?
        def majorityErrorRate = PP.map(row => row.min)
}

trait ConvexHull extends ErrorEstimator {
    def convexHullCoordinates: Vector[Vector[Double]]
}

class PAV(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double]) extends ConvexHull {
    import LinAlgebra._

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
        EvalUtils.minSumProd(PP, pMisspFa)
    }
    
    def bayesErrorRateOld(PP: Matrix): Row = {
        val E = matMul(PP.transpose, pMisspFa)
        val ber = for (err <- E) yield err.min
        ber
    }
    def EER: Double = {
        val objectiveFunction: (Double => Double) = x => {
            val PP: Matrix = logoddsToProbability(Vector(x))
            val minn: Row = EvalUtils.minSumProd(PP, pMisspFa)
            minn(0)
        }
        val maximised = new EvalUtils.BrentOptimizerScalarWrapper(objectiveFunction, priorLogOdds(0), priorLogOdds.last, minimize=false)
        maximised.optimumValue
    }
}


object EvalUtils extends decisions.Shared.LinAlg{
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

    def minSumProd(A: Matrix, B: Matrix): Row = {
        val product = matMul(A.transpose, B)
        val minn = for (row <- product) yield row.min
        minn
    }
}

object Example{
  def main(args: Array[String]): Unit = {
    println("a")
  }
}

