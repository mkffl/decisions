package decisions

import probability_monad._
import scala.util
import java.util.Properties
import collection.JavaConverters._ 

import org.apache.commons.csv.CSVFormat 
import smile.math.MathEx.{logistic, min, log}
import smile.classification._
import smile.data.{DataFrame, Tuple}
import smile.data.formula.Formula
import smile.data.`type`._
import smile.data.measure._
import smile.io.Read
import smile.math.kernel.GaussianKernel

import plotly._, element._, layout._, Plotly._ 

import decisions.TransactionsData._
import decisions.LinAlgebra._
import decisions.SmileKitLearn._
import decisions.SmileFrame._
import decisions.Dataset._
import decisions.Systems._
import decisions.EvalUtils._


/* A full run from data generation to APE-based comparisons
** and risk validation using simulations.
*/
trait CompareSystems extends decisions.Shared.LinAlg 
                        with decisions.Shared.MathHelp
                        with decisions.Shared.FileIO{
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
        title="APE",
        yaxis = Axis(
            range = (0.0, 0.005),
            title = "Error Probability")
    )
        val data = Seq(observedDCFTrace, minDCFTrace, EERTrace, majorityTrace)

        Plotly.plot(s"$plotlyRootP/ape.html", data, layout)
        }
    }

    def prepReliabilityPlot(lo: Array[Double],
                        y: Array[Int]
    ): Tuple2[Array[Double], Array[Double]] = {
        val p = lo.map(logistic)
        val binValue = binnedAccuracy(p, y, binBy0_10)
        val frequency = binValue.map(_.frequency)
        val accuracy = binValue.map(_.accuracy)
        (frequency.toArray, accuracy.toArray)
    }

    def plotReliability(lo1: Array[Double],
                        y1: Array[Int],
                        lo2: Array[Double],
                        y2: Array[Int]
    ) = {
        val (frequency1, accuracy1) = prepReliabilityPlot(lo1, y1)
        val (frequency2, accuracy2) = prepReliabilityPlot(lo2, y2)
        val frequencyPerfect = (0 to 100 by 5).toVector.map(_ / 100.0)
        val accuracyPerfect = frequencyPerfect

        val trace1 = Scatter(
            frequency1.toVector,
            accuracy1.toVector,
            name = "Not Calibrated",
        )
        val trace2 = Scatter(
            frequency2.toVector,
            accuracy2.toVector,
            name = "Calibrated"
        )

        val tracePerfect = Scatter(
            frequencyPerfect,
            accuracyPerfect,
            name = "Perfect Calibration",
            line = Line(dash = Dash.Dot)
        )

        val data = Seq(trace1, trace2, tracePerfect)
        val layout = Layout(
            title="Reliability plot",
            yaxis=Axis(title = "Accuracy"),
            xaxis=Axis(title = "Frequency")
        )

        Plotly.plot(s"$plotlyRootP/reliability.html", data, layout)
        //Plotly.plot("reliability.html", data, layout)
    }

    
    val formula = Formula.lhs("label")

    val plo: Vector[Double] = (BigDecimal(-5.0) to BigDecimal(5.0) by 0.25).map(_.toDouble).toVector
    val rootP = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/transactions_data"    

    def getRecognizer(model: Recognizer, trainDF: DataFrame): (Array[Double] => Double) = model match {
        case m: Logit => LogisticRegression.fit(formula, trainDF).predictProba //TODO: add properties
        case m: RF => RandomForest.fit(formula, trainDF, m.params.getOrElse(new Properties())).predictProba
        case m: SupportVectorMachine => { //TODO: add properties
            val X = formula.x(trainDF).toArray
            val y = formula.y(trainDF).toIntArray.map{case 0 => -1; case 1 => 1}
            val kernel = new GaussianKernel(8.0)
            SVM.fit(X, y, kernel, 5, 1E-3).predictProba
        }            
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
                    yEval: Vector[Int],
                    priorLogOdds: Vector[Double]
    ): APE = {
        //val yEval = evalData.map({case Transaction(usertype, am, cnt) => usertype}).toVector
        val (recog, calib) = spec
        val recogName: String = recog match {case Logit(name, p) => name
                case RF(name, p) => name
                case SupportVectorMachine(name, p) => name
        }
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

    def outcomeCost(Cmiss: Double, Cfa: Double, pred: Int, actual: Int) = pred match {
            case 1 if actual == 0 => Cfa
            case 0 if actual == 1 => Cmiss
            case _ => 0.0
    }

    def DCF(Cmiss: Double, Cfa: Double)(yhat: Vector[Int], y: Vector[Int]): Double = {
        val costs = for {
            (pred, actual) <- yhat.zip(y)
            cost = outcomeCost(Cmiss, Cfa, pred, actual)
            } yield cost
        costs.sum / costs.size.toFloat        
    }
}


object Bug14 extends CompareSystems{
    def temp(args: Array[String]) = {
        val p_w1 = 0.3
        val trainData: Seq[Transact] = transact(p_w1).sample(20000)
        val devData: Seq[Transact] = transact(p_w1).sample(10000)
        val evalData: Seq[Transact] = transact(p_w1).sample(10000)


        val trainSchema = DataTypes.struct(
                new StructField("label", DataTypes.IntegerType),
                new StructField("f1", DataTypes.DoubleType),
                new StructField("f2", DataTypes.DoubleType),
                new StructField("f3", DataTypes.DoubleType),
                new StructField("f4", DataTypes.DoubleType),
                new StructField("f5", DataTypes.DoubleType),
                new StructField("f6", DataTypes.DoubleType),
                new StructField("f7", DataTypes.DoubleType),
                new StructField("f8", DataTypes.DoubleType),
                new StructField("f9", DataTypes.DoubleType),
                new StructField("f10", DataTypes.DoubleType),
        )        

        val trainDF = trainData.toArray.asDataFrame(trainSchema, rootP)
        val xDev = devData.map{case Transact(u,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10) => Array(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10)}.toArray
        val xEval = evalData.map{case Transact(u,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10) => Array(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10)}.toArray
        val yDev = devData.map{case Transact(u,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10) => u}.toArray
        val yEval = evalData.map({case Transact(u,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10) => u}).toVector    

        val prop:  java.util.Map[String, String] = Map("smile.random.forest.trees" -> "100",
            "smile.random.forest.mtry" -> "0",
            "smile.random.forest.split.rule" -> "GINI",
            "smile.random.forest.max.depth" -> "1000",
            "smile.random.forest.max.nodes" -> "10000",
            "smile.random.forest.node.size" -> "2",
            "smile.random.forest.sample.rate" -> "1.0")
        .asJava
        val rfParams = new Properties()
        rfParams.putAll(rfParams)

        val simSpec = (SupportVectorMachine("svm", None), Isotonic("isotonic"))
        val recognizer = getRecognizer(simSpec._1, trainDF)
        val pDev = xDev.map(recognizer)
        val pEvalUncal = xEval.map(recognizer)
        val calibrator = getCalibrator(simSpec._2, pDev, yDev)
        val loEvalCal = pEvalUncal.map(calibrator).map(logit)
        val loEvalUncal = pEvalUncal.map(logit) 
        
        //plotReliability(loEvalUncal, yEval.toArray, loEvalCal, yEval.toArray) // OK
        
        val systems: Seq[Array[Double]] = for (spec <- Seq(simSpec)) yield fitSystem(spec, trainDF, xDev, yDev, xEval)
        val apes: Seq[APE] = for ( (lo, spec) <- systems.zip(Seq(simSpec))) yield evaluateSystem(spec, lo, yEval, plo)
        apes.foreach(plotAPE)
        
        (recognizer, calibrator, apes)
    }

    def more = {
        val (recognizer, calibrator, apes) = temp(Array(""))
        val thetaTest = -0.75
        val ploTest = logistic(thetaTest)
        val thresh = -1.0*thetaTest
        val thresholder: (Double => Int) = lo => if (lo > thresh) {1} else {0}        
        val decisionMaker: (Array[Double] => Int) = recognizer.
                andThen(calibrator).
                andThen(logit).
                andThen(thresholder)

        val verifyData = transact(ploTest).repeat(10)
        /*
        val riskExperiment = for {
            data <- verifyData
            x = data.map{case Transact(u,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10) => Array(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10)}.toArray
            y = data.map{case Transact(u,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10) => u}.toArray
        } yield for {
            (predictor, target) <- x.zip(y)
            prediction = decisionMaker(predictor)
            cost = outcomeCost(1, 1, prediction, target)
        } yield cost
        */
    }
}
