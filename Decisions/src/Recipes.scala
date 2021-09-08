package decisions

import scala.language.postfixOps

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
    import CollectionsStats._

    implicit def floatToDoubleRow(values: Row): Seq[Double] = values.toSeq

    val confidenceBand: Tuple2[Double,Double] => Shape = bounds => new Shape(
        `type`=Some("rect"),
        xref=Some("x"),
        yref=Some("y"),
        x0=Some(s"${bounds._1}"),
        y0=Some(0.0),
        x1=Some(s"${bounds._2}"),
        y1=Some(0.3),
        fillcolor=Some(Color.RGBA(156, 165, 196, 1.0)),
        opacity=Some(0.3),
        line=Some(Line(color = Color.RGBA(156, 165, 196, 1.0), width = 1.0)),
    )

    val expectedLine: Double => Shape = expected => new Shape(
      `type`=Some("line"),
      xref=Some("x"),
      yref=Some("y"),
      x0=Some(s"$expected"),
      y0=Some(0.0),
      x1=Some(s"$expected"),
      y1=Some(0.3),
      fillcolor=None,
      opacity=None,
      line=Some(Line(color = Color.RGBA(55, 128, 191, 1.0), width = 3.0))
    )

    //TODO: rename
    def plotSimulation(observed: Row, expected: Double) = {
        val rect = confidenceBand((observed.percentile(5), observed.percentile(95)))
        val lin = expectedLine(expected)
        val trace = Histogram(observed, histnorm= HistNorm.Probability, name="TBC")
        val layout = Layout().
                     withTitle("Simulation").
                     withYaxis(Axis(title = "y")).
                     withShapes(Seq(rect,lin))
        Plotly.plot(s"$plotlyRootP/simulation.html", Seq(trace), layout)
    }

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

    def plotLLR(scores: Row, llr: Row, minTheta: Double): Unit = {
        val llrTrace = Scatter(
            scores,
            llr,
            name = "Log Likelihood Ratio i.e. log P(x|ω1)/P(x|ω0)",
            mode = ScatterMode(ScatterMode.Lines)
        )
        val minThetaTrace = Scatter(
            scores,
            for (it <- 1 to scores.size) yield minTheta,
            name = "-θ i.e. -log P(ω1)/P(ω0)*Cmiss/Cfa",
            mode = ScatterMode(ScatterMode.Lines, ScatterMode.Markers)
        )
        val layout = Layout(
            title="Scores vs Log Likelihood Ratio",
            yaxis = Axis(
                range = (-5, +5),
                title = "LLR")
        )
        val data = Seq(llrTrace, minThetaTrace)

        Plotly.plot(s"$plotlyRootP/llr.html", data, layout)
    }

    def plotRisk(scores: Row, allRisks: Row): Unit = {
        val risksTrace = Scatter(
            scores,
            allRisks,
            name = "E(r)",
            mode = ScatterMode(ScatterMode.Lines)
        )
        val layout = Layout(
            title="Scores vs Log Likelihood Ratio",
            xaxis = Axis(
                range = (-5, +5),
                title = "Score"),
            yaxis = Axis(
                range = (0, 10),                
                title = "E(r)")
        )
        val data = Seq(risksTrace)

        Plotly.plot(s"$plotlyRootP/risks.html", data, layout)
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

    def plotCCD_LLR_ROC(data: Chart1.ScoreCounts): Unit = {
        val th = data.thresholds
        val correctType: Vector[Double] = data.asCCD.lift(0).getOrElse(th)
        val ccdW0Trace = Bar(th, data.asCCD.lift(0).get).
            withName("Class Conditional Distribition, P(x|ω0)"). 
            withXaxis(AxisReference.X2).
            withYaxis(AxisReference.Y2).
            withMarker(
                Marker().
                withColor(Color.RGB(124, 135, 146)).
                withOpacity(0.5)
            ).
            withWidth(0.5)


        val ccdW1Trace = Bar(th, data.asCCD.lift(1).get).
            withName("Class Conditional Distribition, P(x|ω1)"). 
            withXaxis(AxisReference.X2).
            withYaxis(AxisReference.Y2).
            withMarker(
                Marker().
                withColor(Color.RGB(6, 68, 91)).
                withOpacity(0.8)
            ).
            withWidth(0.5)

        val llrTrace = Scatter(
            th,
            data.asLLR,
            name = "Log-likelihood Ratio P(x|ω1)/P(x|ω0)",
            xaxis = AxisReference.X3,
            yaxis = AxisReference.Y3
        )

        val rocTrace = Scatter(
            data.asROC.lift(0).get,
            data.asROC.lift(1).get,
            name = "Receiving Operator Characteristics P(x>c|ω0) vs P(x>c|ω1)",
            xaxis = AxisReference.X4,
            yaxis = AxisReference.Y4            
        )

        val allPlots = Seq(ccdW0Trace, ccdW1Trace, llrTrace, rocTrace)

        val layout =  Layout(
                title = "Mulitple Custom Sized Subplots",
                width = 700,
                height = 900,
                xaxis = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.Y1),
                    domain = (0, 1)),
                yaxis = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.X1),
                    domain = (0.65, 1)),
                xaxis2 = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.Y2),
                    domain = (0, 1)),
                yaxis2 = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.X2),
                    domain = (0.65, 1)),
                xaxis3 = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.Y3),
                    domain = (0, 1)),
                yaxis3 = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.X3),
                    domain = (0.33, 0.65)),
                xaxis4 = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.Y4),
                    domain = (0, 1)),
                yaxis4 = Axis(
                    anchor = AxisAnchor.Reference(AxisReference.X4),
                    domain = (0, 0.32))
    )

        Plotly.plot(
            s"$plotlyRootP/discimination.html",
            allPlots,
            layout
        )

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
        val trainData: Seq[Transaction] = transact(p_w1).sample(1_000) // Base value is 1_000
        val devData: Seq[Transaction] = transact(p_w1).sample(100_000) // Base value is 100_000
        val evalData: Seq[Transaction] = transact(p_w1).sample(2000) // Base value is 20_000
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

object Chart1 extends CompareSystems{
    import Estimator._
    import CollectionsStats._
    
    // Get train data, assuming balanced labels
    val pa = AppParameters(p_w1=0.5,Cmiss=100,Cfa=5)
    val trainData: Seq[Transaction] = transact(pa.p_w1).sample(1_000)
    val trainDF = trainData.toArray.asDataFrame(trainSchema, rootP)
    // Fit the SVM recognizer
    val baseModel = SupportVectorMachine("svm", None)//RF("rf", None) 
    val recognizer = getRecognizer(baseModel, trainDF)

    // Transform predictions (logit transform) to have scores on ]-inf, +inf[
    // Evaluate CCD on Eval (no dev used) -- Bonus: kernel density
    val loEval = xEval.map(recognizer).map(logit).toVector
    val tarPreds = loEval zip yEval filter{case (lo,y) => y == 1} map {_._1} 
    val nonPreds = loEval zip yEval filter{case (lo,y) => y == 0} map {_._1} 
    def chart1a = plotCCD(nonPreds,tarPreds)
    
    // Fit pav on Eval and plot LLR (line chart)
    val pav = new PAV(loEval, yEval, plodds)
    val (scores, llr) = pav.scoreVSlogodds // scores <-> pavLoEval
    val minTheta = -1 * paramToTheta(pa)
    def chart1b = plotLLR(scores, llr, minTheta)

    // Calculate expected Risk (argmin to find pMissPfa)
    def minPmissPfa(pav: PAV, iScore: Int): Tuple2[Double, Double] = {
        val pMiss = pav.pMisspFa(0).drop(0)
        val pFa = pav.pMisspFa(1).drop(0)
        (pMiss(iScore), pFa(iScore))
    }
    //val iScore = minScoreIndex(llr, minTheta)
    val iScore = llr.map{x => math.abs(x-minTheta) }.argmin
    val (pMiss, pFa) = minPmissPfa(pav, iScore)
    val scoreThreshold = scores(iScore)
    /*  Expected Risk
        E(r) = Cmiss.p(ω1).∫p(x<c|ω1)dx + Cfa.p(ω0).∫p(x>c|ω0)dx
             = Cmiss.p(ω1).Pmiss + Cfa.p(ω0).Pfa
    */
    def expectedRisk(pa: AppParameters)(operatingPoint: Row): Double = pa.p_w1*operatingPoint(0)*pa.Cmiss + (1-pa.p_w1)*operatingPoint(1)*pa.Cfa
    val fraudAppRisk: Row => Double = expectedRisk(pa)(_)
    val allRisks = pav.pMisspFa.transpose.map(p => fraudAppRisk(p))
    val minRisk = fraudAppRisk(Row(pMiss, pFa))
    def chart1c = plotRisk(scores, allRisks)

    // Simulate to verify expected Risk
    // Bonus: Plot expected risk with several thresholds
    // TODO: apply the classifier approach to the estimators
    val thresholder: (Double => User) = score => if (score > scoreThreshold) {Fraudster} else {Regular}
    def classifier:(Array[Double] => User) = recognizer andThen logit andThen thresholder

    def getRisk: Distribution[Double] = for {
        transaction <- transact(pa.p_w1)
        prediction = classifier(transaction.features.toArray)
        risk = cost(pa, transaction.UserType, prediction)
    } yield risk

    def simulate: Distribution[Double] = getRisk.repeat(1000).map(_.sum / 1000.0)
    val simRisk: Row = simulate.sample(200).toVector
    val fithPerc = simRisk.percentile(5)
    val ninetyfithPerc = simRisk.percentile(95)

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

    /* Unit test
        val tarSam = sample(tarPreds,0.02); val nonSam = sample(nonPreds,0.02)
        naiveA(nonSam,tarSam) == smartA(nonSam,tarSam) 
    */
    def sample(data: Row, perc: Double) = {
          require(0 < perc && perc < 1)
          val mask = Distribution.bernoulli(perc).sample(data.size)
          data zip(mask) filter{case (v,m) => m == 1} map(_._1)
    }

    // TODO: Check that U is to test that A = 0.5
    // If so then it's practically useless?
    // check if I can use the estimated variance to compare A1 and A2 for 2 models

    val proportion: Row => Row = counts => {
        val S = counts.sum.toDouble
        counts.map(v => v/S)
    }
    val cumulative: Row => Row = freq => freq.scanLeft(0.0)(_ + _)
    val oneMinus: Row => Row = cdf => cdf.map(v => 1-v)
    val decreasing: Row => Row = data => data.reverse
    val odds: Tuple2[Row,Row] => Row = w0w1 => w0w1._1.zip(w0w1._2).map{case (non,tar) => tar/non}
    val logarithm: Row => Row = values => values.map(math.log)

    /* Bayes Decision Rules
       Plot CCD,LLR and ROC for histogram bins with 2000 points
       - 20 bins for a smooth approximation 
       - Then 100 bins to have a concavity, a segway into PAV
       - PAV counts with 20 bins, LLR and ROC with isocosts to show the link with risk-based decisions
    */

    trait ApproximateDistributions {
        // Define some common operations to estimate distributions
        val pdf = proportion
        val cdf = pdf andThen cumulative
        val rhsArea = cdf andThen oneMinus
        val logodds: Tuple2[Row,Row] => Row = odds andThen logarithm
    }
    case class ScoreCounts(w1Counts: Vector[Double], w0Counts: Vector[Double], thresholds: Vector[Double]) extends ApproximateDistributions {
        val N = w1Counts.size
        require(w0Counts.size == N)
        require(thresholds.size == N)

        /* Present the information */
        def asCCD: Vector[Vector[Double]] = {
            val w0pdf = pdf(w0Counts)
            val w1pdf = pdf(w1Counts)
            Vector(w0pdf,w1pdf)
        }
        def asLLR: Vector[Double] = logodds((pdf(w0Counts),pdf(w1Counts)))
        def asROC: Vector[Vector[Double]] = {
            val fpr = (rhsArea andThen decreasing)(w0Counts)
            val tpr = (rhsArea andThen decreasing)(w1Counts)
            Vector(fpr,tpr)
        }
        def asDET: Vector[Vector[Double]] = ???
        def asPmissPfa: Vector[Vector[Double]] = {
            val pMiss = rhsArea(w1Counts)
            val pFa = rhsArea(w0Counts) // Same as fpr but in the asc order of scores
            Vector(pMiss,pFa)
        }

        /* Given application parameters, return optimal threshold and the corresponding expected risk */
        def minimizeRisk(pa: AppParameters): Tuple2[Double,Double] = ???
        def ber(p_w1: Double): Tuple2[Double,Double] = minimizeRisk(AppParameters(p_w1,1,1))
    }

    val w0Counts = pav.nonTars
    val w1Counts = pav.targets
    val thresholds: Row = pav.pavFit.bins.map(_.getX)
    val pavDist = ScoreCounts(w1Counts,w0Counts,thresholds)
    //plotCCD_LLR_ROC(pavDist)
    val numBins = 100
    val min=loEval.min
    val max=loEval.max
    val w0HistCounts = histogram(nonPreds,numBins,min,max).map(_._2).toVector
    val w1HistCounts = histogram(tarPreds,numBins,min,max).map(_._2).toVector
    val histThresh = histogram(tarPreds,numBins,min,max).map(_._1.toDouble).toVector
    val histDist = ScoreCounts(w1HistCounts,w0HistCounts,histThresh)
    plotCCD_LLR_ROC(histDist)

    // Plot of CCD, LLR and ROC [bins] 
    // then CCD [bins], LLR and ROC [pav]
    // 

}

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

