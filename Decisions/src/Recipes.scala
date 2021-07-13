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
import smile.math.MathEx.{logistic, log}

import plotly._, element._, layout._, Plotly._ 

import decisions._
import java.io._

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
    import utils._
    import decisions.SmileKitLearn._
    import decisions.Dataset._
    import decisions.Systems._ 

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

        def logit(x: Double): Double = {
            log(x/(1-x))
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
        val layout = Layout(
            title="APE"
        )
            val data = Seq(observedDCFTrace, minDCFTrace, EERTrace, majorityTrace)

            Plotly.plot("div-id", data, layout)
            }
        }

        import decisions.utils._

        def prepReliabilityPlot(lo: Array[Double],
                            y: Array[Int]
        ): Tuple2[Array[Double], Array[Double]] = {
            val p = lo.map(logistic)
            val binValue = binnedAccuracy(p, y, binBy0_5)
            val frequency = binValue.map(_.frequency)
            val accuracy = binValue.map(_.accuracy)
            (frequency.toArray, accuracy.toArray)
        }

        def plotReliability(loDev: Array[Double],
                            yDev: Array[Int],
                            loEval: Array[Double],
                            yEval: Array[Int]
        ) = {
        val (devFrequency, devAccuracy) = prepReliabilityPlot(loDev, yDev)
        val (evalFrequency, evalAccuracy) = prepReliabilityPlot(loEval, yEval)

        val devTrace = Scatter(
            devFrequency.toVector,
            devAccuracy.toVector,
            name = "Development",
            mode = ScatterMode(ScatterMode.Lines)
        )
        val evalTrace = Scatter(
            evalFrequency.toVector,
            evalAccuracy.toVector,
            name = "Evaluation",
            mode = ScatterMode(ScatterMode.Lines),
            xaxis = AxisReference.X2,
            yaxis = AxisReference.Y2 
        )
        val data = Seq(devTrace, evalTrace)
        val layout = Layout(
            title="Reliability plot",
            xaxis = Axis(
                anchor = AxisAnchor.Reference(AxisReference.Y1),
                domain = (0, 0.45)),
            yaxis = Axis(
                anchor = AxisAnchor.Reference(AxisReference.X1),
                domain = (0, 1)),
            xaxis2 = Axis(
                anchor = AxisAnchor.Reference(AxisReference.Y2),
                domain = (0.55, 1)),
            yaxis2 = Axis(
                anchor = AxisAnchor.Reference(AxisReference.X2),
                domain = (0, 1))
        )

        Plotly.plot("Reliability", data, layout)
        }
    }

    object Application{
        import utils._
        import decisions.SmileKitLearn._
        import decisions.SmileFrame._     
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

        // Refactor as m = LogisticRegression.fit(formula, train); evalArray.map(m).map(logit)
        def fitTransformRecognizer(spec: System): Array[Double] = spec match {
            // xyTrain = trainData...asDataFrame
            // xEval = evalData.map({case Transaction(usertype, am, cnt) => Array(usertype, am)})
            case (recognizer: Logit, _) => LogisticRegression.fit(formula, train).predictproba(evalArray).map(logit)
            case (recognizer: RF, _) => RandomForest.fit(formula, train).predictproba(evalArray).map(logit)
        }

        def fitTransformCalibrator(spec: System, scores: Array[Double]): Array[Double] = spec match {
            case (_, Uncalibrated) => scores
            /* case Isotonic => {
                val calibScores = m.predictproba(xCalib)
                val calibrator = IsotonicRegression.fit(yCalibScores, calibScores)
                calibrator.predictproba(scores)
            }

            
            */
        }        

        def trainSystem(spec: System): Array[Double] = {
            val recogniserscores = fitTransformRecognizer(spec)
            val calibratorscores = fitTransformCalibrator(spec, recogniserscores)
            calibratorscores
        }

        //TODO: call scores 'lo' for log-odds
        def evaluateSystem(spec: System, scores: Array[Double], priorLogOdds: Vector[Double]): APE = {
            val yEval = evalData.map({case Transaction(usertype, am, cnt) => usertype}).toVector
            val (recog, calib) = spec
            val recogName: String = recog match {case Logit(name) => name; case RF(name) => name}
            val calibName: String = calib match {case Isotonic(name) => name
                case Platt(name) => name
                case Uncalibrated => "Uncalibrated"}
            val steppy = new SteppyCurve(scores.toVector, yEval, priorLogOdds)
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

        val plo: Vector[Double] = (BigDecimal(-5.0) to BigDecimal(5.0) by 0.25).map(_.toDouble).toVector
        val trainedSystems: Seq[Array[Double]] = for (spec <- specs) yield trainSystem(spec)
        val evaluatedSystems: Seq[APE] = for ( (scores, spec) <- trainedSystems.zip(specs)) yield evaluateSystem(spec, scores, plo)
        evaluatedSystems.foreach(plotAPE)
    }

    object AppMap{
        import decisions.SmileFrame._

        val trainData: Seq[Transaction] = transaction.sample(10000)
        val devData: Seq[Transaction] = transaction.sample(2000)
        val evalData: Seq[Transaction] = transaction.sample(2000)

        val trainSchema = DataTypes.struct(
                new StructField("label", DataTypes.IntegerType),
                new StructField("amount", DataTypes.DoubleType),
                new StructField("count", DataTypes.DoubleType),
        )
        
        val formula = Formula.lhs("label")

        val predictSchema = DataTypes.struct(
            new StructField("label", DataTypes.DoubleType),
            new StructField("amount", DataTypes.DoubleType),
            new StructField("count", DataTypes.DoubleType)
        ) 

        val plo: Vector[Double] = (BigDecimal(-5.0) to BigDecimal(5.0) by 0.25).map(_.toDouble).toVector        

        val s1: System = (Logit("loggit"), Uncalibrated)
        val s2: System = (RF("forest"), Uncalibrated)
        val s3: System = (Logit("loggit"), Isotonic("isotonic"))
        val s4: System = (RF("forest"), Isotonic("isotonic"))

        val specs: Seq[System] = List(s1, s2, s3, s4)

        // Prepare datasets
        val rootP = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/transactions_data"
        val trainDF = trainData.toArray.asDataFrame(trainSchema, rootP)
        val xDev = devData.map{case Transaction(u,a,c) => Array(a,c)}.toArray
        val xEval = evalData.map{case Transaction(u,a,c) => Array(a,c)}.toArray
        val yDev = devData.map{case Transaction(u,a,c) => u}.toArray        

        def getRecognizer(model: Recognizer, trainDF: DataFrame): (Array[Double] => Double) = model match {
            case m: Logit => LogisticRegression.fit(formula, trainDF).predictProba
            case m: RF => RandomForest.fit(formula, trainDF).predictProba
            }

        def getCalibrator(model: Calibrator, pDev: Array[Double], yDev: Array[Int]): (Double => Double) = model match {
                case Uncalibrated => x => x
                case c: Isotonic => IsotonicRegressionScaling.fit(pDev, yDev).predict//.predictProba
            }

        /*
        ** y <-> ground truth
        ** x <-> predictors
        ** p <-> predicted probabilities (calibrated or not)
        ** lo <-> log-odds (calibrated or not)
        ** Hence loEvalCal refers to the calibrated log-odds predictions on the evaluation data
        ** TODO: check why the calibrated scores don't show improvements vs uncalibrated
        */
        def fitSystem(spec: System,
                    trainDF: DataFrame, 
                    xDev: Array[Array[Double]],
                    yDev: Array[Int],
                    xEval: Array[Array[Double]]
        ) = {
            // Fit and apply the recognizer
            val recognizer = getRecognizer(spec._1, trainDF)
            val pDev = xDev.map(recognizer)
            val pEvalUncal = xEval.map(recognizer)
            // Fit and apply the calibrator
            val calibrator = getCalibrator(spec._2, pDev, yDev)
            val loEvalCal = pEvalUncal.map(calibrator).map(logit)
            loEvalCal
        }

        def evaluateSystem(spec: System, 
                        lo: Array[Double], 
                        priorLogOdds: Vector[Double]
        ): APE = {
            val yEval = evalData.map({case Transaction(usertype, am, cnt) => usertype}).toVector
            val (recog, calib) = spec
            val recogName: String = recog match {case Logit(name) => name; case RF(name) => name}
            val calibName: String = calib match {case Isotonic(name) => name
                case Platt(name) => name
                case Uncalibrated => "Uncalibrated"}
            val steppy = new SteppyCurve(lo.toVector, yEval, priorLogOdds)
            val pav = new PAV(lo.toVector, yEval, priorLogOdds)
            APE(recogName,
                calibName,
                priorLogOdds,
                steppy.bayesErrorRate,
                pav.bayesErrorRate,
                pav.EER,
                steppy.majorityErrorRate
            )
        }        

        val systems: Seq[Array[Double]] = for (spec <- specs) yield fitSystem(spec, trainDF, xDev, yDev, xEval)
        val apes: Seq[APE] = for ( (scores, spec) <- systems.zip(specs)) yield evaluateSystem(spec, scores, plo)
        apes.foreach(plotAPE) // Why does Isotonic not improve observed DCF?

        // Risk - Experimental
        val simSpec = (RF("forest"), Isotonic("isotonic"))
        val recognizer = getRecognizer(simSpec._1, trainDF)
        val pDev = xDev.map(recognizer)
        val pEvalUncal = xEval.map(recognizer)
        // Fit Calibrator
        val calibrator = getCalibrator(simSpec._2, pDev, yDev)

        val theta = 4.5
        val thresh = -1.0*theta
        val thresholder: (Double => Double) = lo => if (lo > thresh) {1} else {0.0}

        val decisionMaker = recognizer.andThen(calibrator).andThen(logistic).andThen(thresholder)
                                
        val dataset: Distribution[List[Transaction]] = transaction.repeat(5)

        val testtt = for (data <- dataset) yield for (tran <- data.toArray) yield Array(tran.amount, tran.count)
        val rv = for (data <- dataset) yield 
            for {
                tran <- data.toArray
                x = Array(tran.amount, tran.count)
                decision = decisionMaker(x)
            } yield decision





    }
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