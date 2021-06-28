package decisions

import probability_monad._
import scala.util

import org.apache.commons.csv.CSVFormat 
import smile.math.MathEx.{logistic, min}
import smile.classification._
import smile.data.{DataFrame}
import smile.data.formula.Formula
import smile.data.`type`._
import smile.data.measure._
import smile.data.Tuple
import smile.io.Read

import plotly._, element._, layout._, Plotly._ 

import decisions._
import java.io._
import smile.regression

import java.io._

object GaussianTestCase {
    def gaussianScores: Tuple2[Vector[Double], Vector[Int]] = {
        object RepeatableDistribution extends Distributions(new scala.util.Random(54321))

        val targets = for {
            s <- RepeatableDistribution.normal
            //score = logistic(2.0 + logOdds)
            //theta_1 = 2.0 + s
            theta_1 = 2.0 + 2.0*s
            } yield theta_1

        val nonTargets = for {
            s <- RepeatableDistribution.normal
            //score = logistic(-2.0 + logOdds)
            //theta_2 = -2.0 + s
            theta_2 = -2.0 + 2.0*s
            } yield theta_2

        val scores = (targets.sample(1000) ++ nonTargets.sample(100000)).toVector
        val pos: IndexedSeq[Int]  = for (i <- 1 to 1000) yield 1 
        val neg: IndexedSeq[Int]  = for (i <- 1 to 100000) yield 0
        val labels = pos.toVector ++ neg.toVector

        (scores, labels)
    }

    def writeFile(filename: String, lines: Seq[String], headers: String): Unit = {
        val file = new File(filename)
        val bw = new BufferedWriter(new FileWriter(file))
        val linesToPrint = Seq(headers) ++ lines
        for (line <- linesToPrint) {
            bw.write(s"$line\n")
        }
        bw.close()
    }

    val scoresFname = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/unit_data/scores.txt" 
    val labelsFname = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/unit_data/labels.txt"

    val (scores, labels) = gaussianScores
    val ploRange: Vector[Double] = (BigDecimal(-2.0) to BigDecimal(2.0) by 0.5).map(_.toDouble).toVector
    val steppy = new SteppyCurve(scores, labels, ploRange)
    val pav = new PAV(scores, labels, ploRange)   
}



/* A full run from data generation to APE-based comparisons
** and risk validation using simulations.
*/
object CompareSystems{
    import decisions.TransactionsData._
    import decisions.LinAlgebra._



    object utils{
        def writeFile(filename: String, lines: Seq[String], headers: String): Unit = {
            val file = new File(filename)
            val bw = new BufferedWriter(new FileWriter(file))
            val linesToPrint = Seq(headers) ++ lines
            for (line <- linesToPrint) {
                bw.write(s"$line\n")
            }
            bw.close()
        }

        def caseclassToString(tr: Transaction) = tr match {
            case Transaction(la, am, cnt) => f"$la, $am%1.2f, $cnt%1.2f"
            case _ => "not a transaction."
        }

        // TODO: add a _ case to yield a Option[Double]
        def caseclassToNumeric(tr: Transaction): Array[Double] = tr match {
            case Transaction(la, am, cnt) => Array(la.toDouble, am, cnt)
        }



        case class APE(
            val recognizer: String,
            val calibrator: String,
            val priorLogOdds: Vector[Double],
            val observedDCF: Vector[Double],
            val minDCF: Vector[Double],
            val EER: Double,
            val majorityDCF: Vector[Double]
        )

        def plotAPE(data: APE): Unit = data match {
            case APE(recognizer,
            calibrator,
            plo,
            observedDCF,
            minDCF,
            eer,
            majorityDCF) => {          
            val observedDCFTrace = Scatter(
                plo,
                observedDCF,
                name = "Observed DCF",
                mode = ScatterMode(ScatterMode.Lines)
            )
            val minDCFTrace = Scatter(
                plo,
                minDCF,
                name = "minimum DCF",
                mode = ScatterMode(ScatterMode.Lines)
            )
            val EERTrace = Scatter(
                plo,
                for (it <- 1 to plo.size) yield eer,
                name = "EER",
                mode = ScatterMode(ScatterMode.Lines)
            )
            val majorityTrace = Scatter(
                plo,
                majorityDCF,
                name = "Majority Classifier DCF",
                mode = ScatterMode(ScatterMode.Lines)
            )

            val data = Seq(observedDCFTrace, minDCFTrace, EERTrace, majorityTrace)
            val layout = Layout(
                title="APE",
                height = 650,
                width = 600
            )
            Plotly.plot("div-id", data, layout)
            }
        }

    }

    object Application{
        import utils._
        import decisions.SmileKitLearn._
        import decisions.Dataset._
        import decisions.Systems._        

        val trainData = transaction.sample(10000)
        val calibData = transaction.sample(2000)
        val evalData = transaction.sample(2000)
        val evalArray = evalData.map({case Transaction(la, am, cnt) => Array(am, cnt)}).toArray
        
        val rootPath = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/transactions_data"

        writeFile(s"$rootPath/train.csv", trainData.map(caseclassToString), "label,amount,count")
        //writeFile(s"$rootPath/calibration.csv", calibData.map(caseclassToString), "label,amount,count")
        //writeFile("/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/transactions_data/evaluation.csv", evalData.map(caseclassToString), "label,amount,count")

        val schema = DataTypes.struct(
                new StructField("label", DataTypes.IntegerType),
                new StructField("amount", DataTypes.DoubleType),
                new StructField("count", DataTypes.DoubleType),
        )
        val format = CSVFormat.DEFAULT.withFirstRecordAsHeader()
        val train = Read.csv(s"$rootPath/train.csv", format, schema)
        val formula = Formula.lhs("label")
        val predictSchema = DataTypes.struct(
            new StructField("label", DataTypes.DoubleType),
            new StructField("amount", DataTypes.DoubleType),
            new StructField("count", DataTypes.DoubleType)
        ) 

        val s1: System = (Logit("loggit"), Uncalibrated)
        val s2: System = (RF("forest"), Uncalibrated)

        val specs: Seq[System] = List(s1, s2)

        def fitTransformRecognizer(spec: System): Array[Double] = spec match {
            // xyTrain = trainData...asDataFrame
            // xEval = evalData.map({case Transaction(usertype, am, cnt) => Array(usertype, am)})
            case (recognizer: Logit, _) => LogisticRegression.fit(formula, train).predictproba(evalArray)
            case (recognizer: RF, _) => RandomForest.fit(formula, train).predictproba(evalArray)
        }

        def fitTransformCalibrator(spec: System, scores: Array[Double]): Array[Double] = spec match {
            case (_, Uncalibrated) => scores
        }        

        def trainSystem(spec: System): Tuple2[System, Array[Double]] = {
            val recogniserscores = fitTransformRecognizer(spec)
            val calibratorscores = fitTransformCalibrator(spec, recogniserscores)
            (spec, calibratorscores)
        }

        def evaluateSystem(input: Tuple2[System, Array[Double]], priorLogOdds: Vector[Double]): APE = {
            val yEval = evalData.map({case Transaction(usertype, am, cnt) => usertype}).toVector
            val (system, scores) = input
            val (recog, calib) = system
            val recogName: String = recog match {case Logit(name) => name; case RF(name) => name}
            val calibName: String = calib match {case Isotonic(name) => name
                case Platt(name) => name
                case Uncalibrated => "Uncalibrated"}
            val steppy = new SteppyCurve(scores.toVector, yEval, priorLogOdds)  // faulty
            val pav = new PAV(scores.toVector, yEval, priorLogOdds)
            APE(recogName,
                calibName,
                priorLogOdds,
                steppy.bayesErrorRate,
                pav.bayesErrorRate,
                pav.EER,
                steppy.majorityErrorRate
            )
        }

        val plo: Vector[Double] = (BigDecimal(-2.0) to BigDecimal(2.0) by 0.5).map(_.toDouble).toVector
        val trainedSystems: Seq[Tuple2[System, Array[Double]]] = for (spec <- specs) yield trainSystem(spec)
        val evaluatedSystems: Seq[APE] = for (result <- trainedSystems) yield evaluateSystem(result, plo)
        evaluatedSystems.foreach(plotAPE)
}



object ChartRecipes {
    import decisions.LinAlgebra._

    implicit def floatToDoubleRow(values: Row): Seq[Double] = values.toSeq

    def plotCCD(w1Data: Seq[Double], w2Data: Seq[Double], title: String="Class Conditional Densities") = {
        val trace1 = Histogram(w1Data, histnorm= HistNorm.Probability, name="Class ω1")
        val trace2 = Histogram(w2Data, histnorm= HistNorm.Probability, name="Class ω2")
        Seq(trace1, trace2).plot(title=title)
    }

    def plotAPEOld(plo: Row,
                observedDCF: Row, 
                minDCF: Row,
                EER: Double,
                majorityDCF: Row
    ): Unit = {
        val observedDCFTrace = Scatter(
            plo,
            observedDCF,
            name = "Observed DCF",
            mode = ScatterMode(ScatterMode.Lines)
        )
        val minDCFTrace = Scatter(
            plo,
            minDCF,
            name = "minimum DCF",
            mode = ScatterMode(ScatterMode.Lines)
        )
        val EERTrace = Scatter(
            plo,
            for (it <- 1 to plo.size) yield EER,
            name = "EER",
            mode = ScatterMode(ScatterMode.Lines)
        )
        val majorityTrace = Scatter(
            plo,
            majorityDCF,
            name = "Majority Classifier DCF",
            mode = ScatterMode(ScatterMode.Lines)
        )

        val data = Seq(observedDCFTrace, minDCFTrace, EERTrace, majorityTrace)
        val layout = Layout(
            title="APE",
            height = 650,
            width = 600
        )
        Plotly.plot("div-id", data, layout)

        //Seq(observedDCFTrace, minDCFTrace, EERTrace, majorityTrace).plot(title="APE")
    }
}

/*
import decisions.GaussianTestCase._ 
import decisions._ 

val steppy = new SteppyCurve(scores, labels, ploRange) 
val pav = new PAV(scores, labels, ploRange)

val observedDCF = steppy.bayesErrorRate 
val minDCF = pav.bayesErrorRate
val EER = pav.EER
val majorityDCF = steppy.majorityErrorRate
val plo = ploRange

case class APE(
    val recognizer: String,
    val calibrator: String,
    val priorLogOdds: Vector[Double],
    val observedDCF: Vector[Double],
    val minDCF: Vector[Double],
    val EER: Double,
    val majorityDCF: Vector[Double]
)

def makeAPEEstimates(recognizer: String,
                     calibrator: String,
                     plo: Row,
                     scores: Row,
                     labels: Vector[Int]
    ): APE = {
        val steppy = new SteppyCurve(scores, labels, plo) 
        val pav = new PAV(scores, labels, plo)

        APE(recognizer,
                calibrator,
                plo,
                steppy.bayesErrorRate,
                pav.bayesErrorRate,
                pav.EER,
                steppy.majorityErrorRate
        )
    }

Generate and persist training data
Loop thru -> (recogniser, calibrator)
    yield trained model,calibrated model, APE plot




*/