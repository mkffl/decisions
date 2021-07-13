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
import java.io._ // rm
import decisions.Shared.MathHelp


/* A full run from data generation to APE-based comparisons
** and risk validation using simulations.
*/
object CompareSystems extends decisions.Shared.LinAlg with decisions.Shared.MathHelp{
    import decisions.TransactionsData._
    import decisions.LinAlgebra._
    import decisions.SmileKitLearn._
    import decisions.Dataset._
    import decisions.Systems._ 

    implicit def floatToDoubleRow(values: Row): Seq[Double] = values.toSeq

    def plotCCD(w1Data: Seq[Double], w2Data: Seq[Double], title: String="Class Conditional Densities") = {
        val trace1 = Histogram(w1Data, histnorm= HistNorm.Probability, name="Class ω1")
        val trace2 = Histogram(w2Data, histnorm= HistNorm.Probability, name="Class ω2")
        Seq(trace1, trace2).plot(title=title)
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

    def prepReliabilityPlot(lo: Array[Double],
                        y: Array[Int]
    ): Tuple2[Array[Double], Array[Double]] = {
        val p = lo.map(logistic)
        val binValue = binnedAccuracy(p, y, binBy0_05)
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




