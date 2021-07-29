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
//import decisions.Systems._
import decisions.EvalUtils._


/* A full run from data generation to APE-based comparisons
** and risk validation using simulations.
*/
trait CompareSystems extends decisions.Shared.LinAlg 
                        with decisions.Shared.MathHelp
                        with decisions.Shared.FileIO
                        with decisions.Systems{
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
                range = (0.0, 0.2),
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

    /*
    ** y <-> ground truth
    ** x <-> predictors
    ** p <-> predicted probabilities (calibrated or not)
    ** lo <-> log-odds (calibrated or not)
    ** Hence loEvalCal refers to the calibrated log-odds predictions on the evaluation data
    ** TODO: check why the calibrated scores don't show improvements vs uncalibrated
    */

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

    class Estimator(val spec: System){
        import Estimator._
        // Expose eval for audit purposes
        def getyEval = yEval
        def getxEval = xEval
        def getPrevalence = p_w1
        // Fit and apply the recognizer
        val recognizer = getRecognizer(spec._1, trainDF)
        val pDev = xDev.map(recognizer)
        val pEvalUncal = xEval.map(recognizer)
        // Fit and apply the calibrator
        val calibrator = getCalibrator(spec._2, pDev, yDev)
        val loEvalCal = pEvalUncal.map(calibrator).map(logit)
        // Evaluate the system
        val steppy = new SteppyCurve(loEvalCal.toVector, yEval, plodds)
        val pav = new PAV(loEvalCal.toVector, yEval, plodds)
        // Plot the results
        def getAPE: APE = APE(getName(spec._1),
            getName(spec._2),
            plodds,
            steppy.bayesErrorRate,
            pav.bayesErrorRate,
            pav.EER,
            steppy.majorityErrorRate
        )
    }

    object Estimator{
        // Instantiate the data shared across estimators
        val p_w1 = 0.3
        val trainData: Seq[Transaction] = transact(p_w1).sample(1_000)
        val devData: Seq[Transaction] = transact(p_w1).sample(100_000)
        val evalData: Seq[Transaction] = transact(p_w1).sample(20_000)
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
        def extractFeatures(row: Transaction): Array[Double] = row match {
            case Transaction(u,feats) => feats.toArray
        }
        def extractUser(row: Transaction): Int = row match {
            case Transaction(Regular,feats) => 0
            case Transaction(Fraudster,feats) => 1
        }

        val trainDF = trainData.toArray.asDataFrame(trainSchema, rootP)
        val xDev = devData.map(extractFeatures).toArray
        val xEval = evalData.map(extractFeatures).toArray
        val yDev = devData.map(extractUser).toArray
        val yEval = evalData.map(extractUser).toVector
        val plodds: Vector[Double] = (BigDecimal(-5.0) to BigDecimal(5.0) by 0.25).map(_.toDouble).toVector

        // Methods used during evaluation
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
        def getName(tr: Transformer) = tr match {
                case Logit(name, p) => name
                case RF(name, p) => name
                case SupportVectorMachine(name, p) => name
                case Isotonic(name) => name
                case Platt(name) => name
                case Uncalibrated => "Uncalibrated"
                case _ => "Not referenced"
        }
    }

}

/*
import decisions.Systems._; import decisions.CompareSystems._; val calibExp = new Estimator( ( SupportVectorMachine("svm", None), Isotonic("isotonic")) ) 
val uncalibExp = new Estimator( ( SupportVectorMachine("svm", None), Uncalibrated) )

*/

object Bug14 extends CompareSystems{
    def modelCheck = {
        // one calib and one non calibrated model
        // assert that calibrated scores are monotonous
        //Note: svm returns negative probabilities

        val prop:  java.util.Map[String, String] = Map("smile.random.forest.trees" -> "100",
            "smile.random.forest.mtry" -> "0",
            "smile.random.forest.split.rule" -> "GINI",
            "smile.random.forest.max.depth" -> "1000",
            "smile.random.forest.max.nodes" -> "10000",
            "smile.random.forest.node.size" -> "2",
            "smile.random.forest.sample.rate" -> "1.0").asJava
        val rfParams = new Properties()
        rfParams.putAll(rfParams)

        val calibExp = new Estimator( ( SupportVectorMachine("svm", None), Isotonic("isotonic")) ) 
        val uncalibExp = new Estimator( ( SupportVectorMachine("svm", None), Uncalibrated) )
        (calibExp, uncalibExp)
        //import decisions.Bug14._; val (calibExp, uncalibExp) = modelCheck
        // not monotonous... investigate
    }

    def pavCheck = {
        import decisions.TransactionsData._ 

        def getLogodds(label: Int) = label match {
            case 0 => Distribution.normal
            case 1 => Distribution.normal*1.5 + 2.0
        }
        def lodds = for {
            label <- Distribution.bernoulli(0.5)
            lo <- getLogodds(label)
        } yield (label, lo)
        def monot_func(x: Double) = 1/(1+scala.math.exp(-1.5*x+2))

        val data = lodds.sample(2000)
        val lo = data.map(_._2).toVector
        val loUncalib = data.map(_._2).map(monot_func).toVector
        val y = data.map(_._1).toVector

        val pavCalib = new PAV(lo, y, Estimator.plodds)
        val pavUncalib = new PAV(loUncalib, y, Estimator.plodds)
        (pavCalib, pavUncalib, data)
        //import decisions.Bug14._; val (pavCalib, pavUncalib, data) = pavCheck
    }
    class Experiment(val e: Estimator,
                     var p_w1: Double, 
                     var Cmiss: Double, 
                     var Cfa: Double){
        import Experiment._

        def theta = getTheta(p_w1, Cmiss, Cfa)
        def thresholder: (Double => User) = lo => if (lo > -1*theta) {Fraudster} else {Regular}

        def decisionMaker:(Array[Double] => User) = observation =>
                e.recognizer.andThen(e.calibrator).andThen(logit).andThen(thresholder)(observation)

        def simulate(nsamples: Int = 1_000): Distribution[Double] = for {
                sample <- transact(p_w1).repeat(nsamples) // Asume true p_w1 == modeler's belief
                decisions = sample.map(_.features.toArray).map(decisionMaker)
                preds = sample.map(_.UserType).zip(decisions).map{case (a,b) => Decision(a,b)}
            } yield pError(preds)
    }

    object Experiment{
        case class Decision(userType: User, decision: User)
        def pError(d: Seq[Decision]): Double = d.count(o => o.userType != o.decision).toDouble / d.size

        def getTheta(p_w1: Double, Cmiss: Double, Cfa: Double) = log(p_w1/(1-p_w1)*(Cmiss/Cfa))

        def apply(e: Estimator,
                     p_w1: Double, 
                     Cmiss: Double, 
                     Cfa: Double): Experiment = {
                         val ex = new Experiment(e,p_w1,Cmiss,Cfa)
                         ex
                     }
    }

    // val ex = Experiment(calibExp, 0.2, 100, 5) // theta = 1.609

    object Reconciliation extends decisions.Shared.MathHelp{
        import Experiment._

        def getClosest(num: Double, listNums: Vector[Double]) =
            listNums.minBy(v => math.abs(v - num))
        def constructThreshold(theta: Double)(lo: Double): User = if (lo > -1*theta) {Fraudster} else {Regular}
        def loModel:(Array[Double] => Double) = observation =>
                calibExp.recognizer.andThen(calibExp.calibrator).andThen(logit)(observation)
        val (calibExp, uncalibExp) = modelCheck
        
        // Application parameters
        val p_w1=0.3; val Cmiss=100; val Cfa=5;
        val ex = Experiment(calibExp, p_w1, Cmiss, Cfa)
        val theta = getTheta(p_w1, Cmiss, Cfa)
        val p_tilde_w1 = logistic(theta) // sample with this prior

        val xData = transact(p_tilde_w1).sample(1_000)
        val xEval = xData.map(_.features.toArray).toArray //calibExp.getxEval
        val yEval = xData.map(_.UserType)//calibExp.getyEval
        
        val targetPrErr: Double = calibExp.getAPE.priorLogOdds.zip(calibExp.getAPE.observedDCF).minBy(tup => math.abs(tup._1-theta))._2

        val loPreds = xEval.map(loModel)
        val thresholder = constructThreshold(theta)(_)
        
        def getPmissPfa(theta: Double, lo: Vector[Double], labels: Vector[User]): Tuple2[Double,Double] = {
            val thr = -1*theta
            val tar = lo.zip(labels).filter(_._2==Fraudster).map(_._1)
            val non = lo.zip(labels).filter(_._2==Regular).map(_._1)
            val pMiss = tar.count(v => thr > v)/tar.size.toDouble
            val pFa = non.count(v => thr < v)/non.size.toDouble
            (pMiss, pFa)
        }

        val preds = loPreds.map(x => thresholder(x)).toVector

        def prErr(theta: Double, pMisspFa: Tuple2[Double,Double]) = logistic(theta)*pMisspFa._1 + logistic(-theta)*pMisspFa._2  //using discrimination points

        def accuracy(labels: Vector[User], hardPredictions: Vector[User]) = 
            labels.zip(preds).count(tup => tup._1 != tup._2).toDouble / labels.size.toDouble //using False counts

        val operatingPoint = getPmissPfa(theta, loPreds.toVector, yEval.toVector)
        val check = prErr(theta, operatingPoint)
        val acc = accuracy(yEval.toVector, preds)

    }
}
