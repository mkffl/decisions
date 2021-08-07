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
                        with decisions.Systems
                        with decisions.Shared.Validation{
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

    // priors should be probabilities for easier interpretation
    def plotRisks(priors: Vector[Double], risks: Vector[Double]): Unit = {
        val riskTrace = Scatter(
            priors,
            risks,
            name = "Estimated Risks",
            mode = ScatterMode(ScatterMode.Lines)
        )
        val layout = Layout(
            title="Risks",
            yaxis = Axis(
                title = "Expected Risks")
        )
        val data = Seq(riskTrace)

        Plotly.plot(s"$plotlyRootP/expected_risk.html", data, layout)        
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
        // Expose models
        def pCalibrated:(Array[Double] => Double) = recognizer andThen calibrator
        def loCalibrated:(Array[Double] => Double) = pCalibrated andThen logit
        def hardCalibrated(theta: Double):(Array[Double] => User) = {
            val thresholder: (Double => User) = lo => if (lo > -1*theta) {Fraudster} else {Regular}
            loCalibrated andThen thresholder        
        }
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

/** Represents attempts to validate the observed Bayes Error with simuations using the
  * the true data generation process for chosen application parameters.
  * 
  * {{
        val calibrated = new Estimator((SupportVectorMachine("svm", None), Isotonic("isotonic")))
        val pa = AppParameters(0.3,100,5)
        val ex = Reconciliation(calibrated, pa)
        ex.simulate(500).hist // Observe simulation results
        ex.inferRiskFromAPE // Compare with APE data
  * }}
  * 
  * @param e APE estimator that includes the trained system and the BER estimates
  * @param p Application parameter inputs used for validation
  */
    class Reconciliation(val e: Estimator,
                         var p: AppParameters
    ){
        import Reconciliation._

        val theta = paramToTheta(p)
        val decisionMaker:(Array[Double] => User) = e.hardCalibrated(theta)

        def simulate(nsamples: Int = 1_000): Distribution[Double] = for {
                sample <- transact(p.p_w1).repeat(nsamples) // Asume true p_w1 == modeler's belief
                predictions = sample.map(_.features.toArray).map(decisionMaker)
                totalCost = sample.map(_.UserType).zip(predictions).map(tup => cost(p,tup._1,tup._2)).reduceLeft(_+_)
            } yield totalCost / nsamples.toDouble

        def inferRiskFromAPE: Double = {
            val priorRisk = getPriorRisk(p)
            val i = getClosestIndex(e.getAPE.priorLogOdds, theta)
            val targetPrErr = e.getAPE.observedDCF(i)
            targetPrErr * priorRisk
        }            
    }

    object Reconciliation{
        def paramToTheta(p: AppParameters): Double = log(p.p_w1/(1-p.p_w1)*(p.Cmiss/p.Cfa))

        def getPriorRisk(p: AppParameters): Double = p.p_w1*p.Cmiss + (1-p.p_w1)*p.Cfa

        /** Find index of closest number from the target in a list
          * {{
          * val target = 3.2
          * val nums = List(-2.0, 3.0, 4.0)
          * getClosestIndex(nums, target) // returns 1
          * }}
          */
        def getClosestIndex(nums: Seq[Double], target: Double): Integer = 
            nums.zipWithIndex.minBy(tup => math.abs(tup._1-target))._2

        def apply(e: Estimator,
                     p: AppParameters): Reconciliation = {
                         val ex = new Reconciliation(e,p)
                         ex
        }
    }
    object Reconciliate {
        import Reconciliation._


        // move to evaluations
        def prErr(plo: Double, pMisspFa: Tuple2[Double,Double]) = plo*pMisspFa._1 + (1-plo)*pMisspFa._2  //using discrimination points
        // move to evaluations
        def accuracy(labels: Vector[User], hardPredictions: Vector[User]) = 
            labels.zip(hardPredictions).count(tup => tup._1 != tup._2).toDouble / labels.size.toDouble //using False counts        

        def getPmissPfa(theta: Double, lo: Vector[Double], labels: Vector[User]): Tuple2[Double,Double] = {
            val thr = -1*theta
            val tar = lo.zip(labels).filter(_._2==Fraudster).map(_._1)
            val non = lo.zip(labels).filter(_._2==Regular).map(_._1)
            val pMiss = tar.count(v => thr > v)/tar.size.toDouble
            val pFa = non.count(v => thr < v)/non.size.toDouble
            (pMiss, pFa)
        }
      

        // 1. Load model
        val calibrated = new Estimator((SupportVectorMachine("svm", None), Isotonic("isotonic")))
        val pa = AppParameters(0.3,100,5)
        val ex = Reconciliation(calibrated, pa)

        def makeParams(pa: Tuple3[Double,Double,Double]): AppParameters = AppParameters(pa._1,pa._2,pa._3)

        def getRisks(priors: Vector[Double], Cmiss: Double, Cfa: Double, plos: Vector[Double], dcf: Vector[Double]) = {
            val pr = for (prior <- priors) yield AppParameters(prior, Cmiss, Cfa)
            val thetas = pr map paramToTheta
            val prRisks = pr map getPriorRisk
            val is: Vector[Integer] = thetas.map(t => getClosestIndex(plos, t))
            val targets = is.map(index => dcf(index))
            targets zip prRisks map Function.tupled(_*_)
        }

        object plotRisks{
            val priors: Vector[Double] = ???
            val Cmiss = 100
            val Cfa = 5
            val plos: Vector[Double] = ???
            val bers: Vector[Double] = ???

        }
        /*
        pr => thetas => map getClosestPlo => dcf.apply

        val preds = loPreds.map(x => thresholder(x)).toVector

        val operatingPoint = getPmissPfa(theta, loPreds.toVector, yEval.toVector)
        val check = prErr(theta, operatingPoint)
        val acc = accuracy(yEval.toVector, preds)
        */

    }
}
