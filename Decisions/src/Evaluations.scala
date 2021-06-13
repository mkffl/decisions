package decisions
import plotly._, element._, layout._, Plotly._ 
import collection.JavaConverters._
import be.cylab.java.roc.Roc
import scala.util
import smile.classification._
import smile.math.MathEx.{logistic, min}

// An ErrorEstimator object stores all metrics for a pair of recognizer/calibrator
trait ErrorEstimator {
    // add a forall check for the vector lengths
    val scores: Vector[Double]
    val labels: Vector[Int]
    val priorLogOdds: Vector[Double]
    val recognizer: String
    val calibrator: String
    val name: String
    def summary: String = s"$name using $recognizer and $calibrator."

}

class Observed(val scores: Vector[Double],
    val labels: Vector[Int],
    val priorLogOdds: Vector[Double],
    val recognizer: String,
    val calibrator: String    
) extends ErrorEstimator {
    val name = "Observed Bayes Error rate"
    def observed: Vector[Double] = utils.observedBER(scores, labels, priorLogOdds)
}

class Majority(val scores: Vector[Double],
    val labels: Vector[Int],
    val priorLogOdds: Vector[Double],
    val recognizer: String,
    val calibrator: String    
) extends ErrorEstimator {
    val name = "Majority classifier error rate"
    def majority: Vector[Double] = utils.majorityErrorRate(scores, labels, priorLogOdds)
}    

class Optimal(val scores: Vector[Double],
    val labels: Vector[Int],
    val priorLogOdds: Vector[Double],
    val recognizer: String,
    val calibrator: String    
) extends ErrorEstimator {
    val name = "Optimal Bayes error rate"
    def optimal: Vector[Double] = ???
}

class EER(val scores: Vector[Double],
    val labels: Vector[Int],
    val priorLogOdds: Vector[Double],
    val recognizer: String,
    val calibrator: String    
) extends ErrorEstimator {
    val name = "Expected Error Rate"
    def EER: Vector[Double] = ???
}

object utils {
    def ordinalRank(arr: Seq[Double]): Seq[Int] = {
        val withIndices = arr.zipWithIndex
        val valueSorted = withIndices.sortBy(_._1)
        val withRank = valueSorted.zipWithIndex
        val indexSorted = withRank.sortBy(_._1._2)
        for((valueAndIndexTuple, rank) <- indexSorted) yield rank + 1
    }

    def pMisspFaPoints(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double]): Tuple2[Vector[Double], Vector[Double]] = {
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
        (pMiss.map(_.toDouble).toVector, pFa.map(_.toDouble).toVector)
    }
    
    def oddsToProba(priorLogOdds: Vector[Double]): Tuple2[Vector[Double], Vector[Double]] = {
        val pTar: Vector[Double] = priorLogOdds.map(logistic)
        val pNon: Vector[Double] = priorLogOdds.map(x => -logistic(x))
        (pTar, pNon)
    }

    // TODO: unit test against PYLLR results
    def observedBER(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double]) = {
        val (pTar, pNon) = oddsToProba(priorLogOdds)
        val (pMiss, pFa) = pMisspFaPoints(scores, labels, priorLogOdds)
        val ber = for ((((pTar,pMiss),pNon),pFa) <- (pTar zip pMiss zip pNon zip pFa) ) yield pTar * pMiss + pNon * pFa
        ber
    }

    def majorityErrorRate(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double]) = {
        val (pTar, pNon) = oddsToProba(priorLogOdds)
        val (pMiss, pFa) = pMisspFaPoints(scores: Vector[Double], labels: Vector[Int], priorLogOdds: Vector[Double])
        val majority = for ((ptar, pnon) <- pTar zip pNon) yield min(ptar, pnon,100.0) // Smile's min method needs 3 or more inputs
        majority
    }
}


object Likelihood{

    def plotCCD(w1Data: Seq[Double], w2Data: Seq[Double], title: String="Class Conditional Densities") = {
        val trace1 = Histogram(w1Data, histnorm= HistNorm.Probability, name="Class ω1")
        val trace2 = Histogram(w2Data, histnorm= HistNorm.Probability, name="Class ω2")
        Seq(trace1, trace2).plot(title=title)
    }

    type Coord = be.cylab.java.roc.CurveCoordinates 

}




object Example{
  def main(args: Array[String]): Unit = {
    // 1. Assessment making with APE and ROC
    // 1.a. Get data sets
    /*
    val tr = ???
    val cal = ???
    val dev = ???
    */

    // 1.b. Fit the estimators i.e. recognizer and calibrator
    /*
    val rfRecog = ???
    val logisticRecog = ???
    val logisticCalib = ???
    val isotonicCalib = ???
    val identityCalib = ???
    */

    // 2. Validation with distributions

    // Need plotting recipes
println("a")
  }
}

/*
import plotly._, element._, layout._, Plotly._ 
import scala.util
import smile.classification._

import collection.JavaConverters._

object LLR {
}
*/