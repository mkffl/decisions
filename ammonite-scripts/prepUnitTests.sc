import probability_monad._
import smile.classification._
import smile.math.MathEx.{logistic, min}
import scala.util

import decisions._
import java.io._

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
val (scores, labels) = gaussianScores
val ploRange: Vector[Double] = (BigDecimal(-2.0) to BigDecimal(2.0) by 0.5).map(_.toDouble).toVector
val steppy = new SteppyCurve(scores, labels, ploRange)

def writeFile(filename: String, lines: Seq[String]): Unit = {
    val file = new File(filename)
    val bw = new BufferedWriter(new FileWriter(file))
    for (line <- lines) {
        bw.write(s"$line\n")
    }
    bw.close()
}

val scoresFname = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/unit_data/scores.txt" 
val labelsFname = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/unit_data/labels.txt"
writeFile(scoresFname, scores.map(_.toString))
writeFile(labelsFname, labels.map(_.toString))




