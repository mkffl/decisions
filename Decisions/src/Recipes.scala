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

import decisions.TransactionsData._, AUC._
import decisions.SmileKitLearn._
import decisions.SmileFrame._
import decisions.Dataset._
//import decisions.Systems._
import decisions.EvalUtils._


/* A full run from data generation to APE-based comparisons
** and risk validation using simulations.
*/

/*
object CompareSystems extends decisions.Shared.LinAlg 
                        with decisions.Shared.MathHelp
                        with decisions.Shared.FileIO
                        with decisions.Systems
                        with decisions.Shared.Validation{
    import CollectionsStats._

    implicit def floatToDoubleRow(values: Row): Seq[Double] = values.toSeq

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

    

    
    ** y <-> ground truth
    ** x <-> predictors
    ** p <-> predicted probabilities (calibrated or not)
    ** lo <-> log-odds (calibrated or not)
    ** Hence loEvalCal refers to the calibrated log-odds predictions on the evaluation data
    ** TODO: check why the calibrated scores don't show improvements vs uncalibrated
    */
    /*

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


        // Methods used during evaluation
        
    }
*/

object Recipes extends decisions.Shared.LinAlg 
                        with decisions.Shared.MathHelp
                        with decisions.Shared.FileIO
                        with decisions.Systems
                        with decisions.Shared.Validation{
    import CollectionsStats._
    import MatLinAlg._, RowLinAlg._

    object Data{
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
    }
    object FitSystems{
        val formula = Formula.lhs("label")

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


    // Plotly recipes
    object Plots{
        import Evals._


        def annotate(x: Double,
                y: Double,
                xref: Int,
                yref: Int,
                text: String,
                showarrow: Boolean = false
        ): Annotation = new Annotation().
                withXref(xref match {case 1 => Ref.Axis(AxisReference.X1)
                        case 2 => Ref.Axis(AxisReference.X2)
                        case 3 => Ref.Axis(AxisReference.X3)
                        case 4 => Ref.Axis(AxisReference.X4)
                }).
                withYref(xref match {case 1 => Ref.Axis(AxisReference.Y1)
                        case 2 => Ref.Axis(AxisReference.Y2)
                        case 3 => Ref.Axis(AxisReference.Y3)
                        case 4 => Ref.Axis(AxisReference.Y4)
                }).
                withX(x).
                withY(y).
                withText(text).
                withShowarrow(showarrow)

            //xref = xref match {case 1 => }
        //)

        def lineShape(x0: Double,
                    y0: Double,
                    x1: Double,
                    y1: Double,
                    xref: String = "x",
                    yref: String = "y",
                    c: Seq[Int] = Seq(129,2,2,1)
        ): Shape = new Shape(
                `type`=Some("line"),
                xref=Some(xref),
                yref=Some(yref),
                x0=Some(s"$x0"),
                y0=Some(y0),
                x1=Some(s"$x1"),
                y1=Some(y1),
                fillcolor=None,
                opacity=None,
                line=Some(Line(color=Color.RGBA(c(0), c(1), c(2), c(3)), width = 1.0,dash=Dash.Dot))
        )

        def rectangleShape(x0: Double,
                    y0: Double,
                    x1: Double,
                    y1: Double,
                    xref: String = "x",
                    yref: String = "y",
                    c: Seq[Int] = Seq(156, 165, 196, 1)
        ): Shape = new Shape(
            `type`=Some("rect"),
            xref=Some("x"),
            yref=Some("y"),
            x0=Some(s"$x0"),
            y0=Some(y0),
            x1=Some(s"$x1"),
            y1=Some(y1),
            fillcolor=Some(Color.RGBA(c(0), c(1), c(2), c(3))),
            opacity=Some(0.3),
            line=Some(Line(color = Color.RGBA(c(0), c(1), c(2), c(3)), width = 1.0)),
        )        

        def plotCCD(w0pdf: Row,w1pdf: Row,thresholds: Row, fName: String) = {
            val ccdW0Trace = Bar(thresholds, w0pdf).
                withName("P(s|ω0)"). 
                withMarker(
                    Marker().
                    withColor(Color.RGB(124, 135, 146)).
                    withOpacity(0.5)
                ).
                withWidth(0.15)
    
    
            val ccdW1Trace = Bar(thresholds, w1pdf).
                withName("P(s|ω1)"). 
                withMarker(
                    Marker().
                    withColor(Color.RGB(6, 68, 91)).
                    withOpacity(0.8)
                ).
                withWidth(0.15)
                
            val traces = Seq(ccdW0Trace,ccdW1Trace)
            
            val layout = Layout().
                    withTitle("Class Conditional Distributions (CCD)").
                    withWidth(900).
                    withHeight(700).                    
                    withXaxis(Axis(title="s",range=(-5,+5))).
                    withYaxis(Axis(title="p(s|w_i)"))

            Plotly.plot(s"$plotlyRootP/$fName-ccd.html", traces, layout)
        }

        def plotCCD_LLR_E_r(w0pdf: Row,
                    w1pdf: Row, 
                    llr: Row, 
                    e_r: Row,
                    thresholds: Row,
                    cutLine1: Segment,
                    cutLine2: Segment,
                    cutLine3: Segment,
                    thetaLine: Segment,
                    minRLine: Segment,
                    fName: String
        ) = {
            val ccdW0Trace = Bar(thresholds, w0pdf).
                withName("P(s|ω0)"). 
                withMarker(
                    Marker().
                    withColor(Color.RGB(124, 135, 146)).
                    withOpacity(0.5)
                ).
                withWidth(0.5).
                withXaxis(AxisReference.X1).
                withYaxis(AxisReference.Y1)                


            val ccdW1Trace = Bar(thresholds, w1pdf).
                withName("P(s|ω1)"). 
                withMarker(
                    Marker().
                    withColor(Color.RGB(6, 68, 91)).
                    withOpacity(0.8)
                ).
                withWidth(0.5).
                withXaxis(AxisReference.X1).
                withYaxis(AxisReference.Y1)               

            val llrTrace = Scatter(thresholds, llr).
                withName("Log Likelihood Ratio").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X2).
                withYaxis(AxisReference.Y2) 

            val e_rTrace = Scatter(thresholds, e_r).
                withName("Expected risk").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X3).
                withYaxis(AxisReference.Y3)                 

            val traces = Seq(ccdW0Trace,ccdW1Trace,llrTrace,e_rTrace)

            val cutOff1 = cutLine1 match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x1","y1")}
            val cutOff2 = cutLine2 match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x2","y2")}
            val cutOff3 = cutLine3 match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x3","y3")}
            val llrminTheta = thetaLine match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x2","y2")}
            val minRisk = minRLine match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x3","y3")}            

            val shapes = Seq(cutOff1,cutOff2,cutOff3,llrminTheta,minRisk)

            val layout =  Layout().
                    withTitle("Bayes Decisions").
                    withWidth(1000).
                    withHeight(900).
                    withXaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.Y1),domain=(0, 1),range=(-5,+5),title="score (s)")).
                    withYaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.X1),domain=(0.65, 1),title="P(s|ωi)")).
                    withXaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.Y2),domain=(0, 1),range=(-5,+5),title="score (s)")).
                    withYaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.X2),domain=(0.33, 0.65),title="llr(ω0)")).
                    withXaxis3(Axis(anchor=AxisAnchor.Reference(AxisReference.Y3),domain=(0, 1),title="score (s)")).
                    withYaxis3(Axis(anchor=AxisAnchor.Reference(AxisReference.X3),domain=(0, 0.32),range=(-5,+5),title="E(risk)")).
                    withShapes(shapes)

            Plotly.plot(s"$plotlyRootP/$fName-bayesdecisions1.html", traces, layout)
        }

        def plotUnivarHist(observed: Row, title: String, xtitle: String, vlines: Option[Seq[Segment]], confidence: Option[Segment], fName: String) = {

            val conf: Option[Seq[Shape]] = confidence.map{case Segment(Point(x0,y0),Point(x1,y1)) => Seq(rectangleShape(x0,y0,x1,y1))}

            val lin: Option[Seq[Shape]] = vlines.map(xs => xs.map{case Segment(Point(x0,y0),Point(x1,y1)) => lineShape(x0,y0,x1,y1)})

            val shapes: Option[Seq[Shape]] = for {
                c <- conf
                l <- lin
            } yield c ++ l
 
            val trace = Histogram(observed, name=title, histnorm = HistNorm.ProbabilityDensity)

            val layout = Layout().
                    withTitle(title).
                    withXaxis(Axis(title=xtitle)).
                    withYaxis(Axis(title="Frequency")).
                    withShapes(shapes)

            Plotly.plot(s"$plotlyRootP/$fName-simulation.html", Seq(trace), layout)
        }

        def plotCCD_LLR_ROC(w0pdf: Row,
                    w1pdf: Row, 
                    llr: Row, 
                    fpr: Row,
                    tpr: Row,
                    thresholds: Row,
                    cutLine: Segment,
                    thetaLine: Segment,
                    rocLine: Segment, 
                    fName: String
        ) = {
            val ccdW0Trace = Bar(thresholds, w0pdf).
                withName("P(s|ω0)"). 
                withMarker(
                    Marker().
                    withColor(Color.RGB(124, 135, 146)).
                    withOpacity(0.5)
                ).
                withWidth(0.5).
                withXaxis(AxisReference.X1).
                withYaxis(AxisReference.Y1)                

            val ccdW1Trace = Bar(thresholds, w1pdf).
                withName("P(s|ω1)"). 
                withMarker(
                    Marker().
                    withColor(Color.RGB(6, 68, 91)).
                    withOpacity(0.8)
                ).
                withWidth(0.5).
                withXaxis(AxisReference.X1).
                withYaxis(AxisReference.Y1)                 

            val llrTrace = Scatter(thresholds, llr).
                withName("Log Likelihood Ratio").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X2).
                withYaxis(AxisReference.Y2) 

            val rocTrace = Scatter(fpr, tpr).
                withName("Receiving Operator Characteristics (ROC)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X3).
                withYaxis(AxisReference.Y3)                 

            val traces = Seq(ccdW0Trace,ccdW1Trace,llrTrace,rocTrace)

            val ccdCutOff = cutLine match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x1","y1")}
            val llrminTheta = thetaLine match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x2","y2")}
            val isocost = rocLine match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x3","y3")}

            val shapes = Seq(ccdCutOff,llrminTheta,isocost)

            val layout =  Layout().
                    withTitle("Bayes Decisions with the ROC").
                    withWidth(1000).
                    withHeight(900).
                    withXaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.Y1),domain=(0, 1),range=(-5,+5),title="score (s)")).
                    withYaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.X1),domain=(0.65, 1),title="P(s|ωi)")).
                    withXaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.Y2),domain=(0, 1),range=(-5,+5),title="score (s)")).
                    withYaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.X2),domain=(0.33, 0.65),title="log-likelihood ratio")).
                    withXaxis3(Axis(anchor=AxisAnchor.Reference(AxisReference.Y3),domain=(0, 1),range=(-0.05,0.6),title="False Positive Rate")).
                    withYaxis3(Axis(anchor=AxisAnchor.Reference(AxisReference.X3),domain=(0, 0.32),range=(0,1.05),title="True Positive Rate")).
                    withShapes(shapes)

            Plotly.plot(s"$plotlyRootP/$fName-bayesdecisions2.html", traces, layout)
        }

        def plotLLR_ROC(llr: Row,
                    llrPAV: Row,        
                    fpr: Row,
                    tpr: Row,
                    fprPAV: Row,
                    tprPAV: Row,                    
                    thresholds: Row,
                    thresholdsPAV: Row, 
                    fName: String
        ) = {
            val llrTrace = Scatter(thresholds, llr).
                withName("LLR (histogram)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X1).
                withYaxis(AxisReference.Y1)
                
            val llrPAVTrace = Scatter(thresholdsPAV, llrPAV).
                withName("LLR (PAV)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X1).
                withYaxis(AxisReference.Y1)                 

            val rocTrace = Scatter(fpr, tpr).
                withName("ROC (histogram)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X2).
                withYaxis(AxisReference.Y2)

            val rocPAVTrace = Scatter(fprPAV, tprPAV).
                withName("ROC (PAV)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X2).
                withYaxis(AxisReference.Y2)                

            val traces = Seq(llrTrace,llrPAVTrace,rocTrace,rocPAVTrace)

            val layout =  Layout().
                    withTitle("LLR and ROC curves - Histogram vs PAV").
                    withWidth(1000).
                    withHeight(900).
                    withXaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.Y1),domain=(0, 1),range=(-5,+5),title="score (s)")).
                    withYaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.X1),domain=(0.52, 1),title="Log-likelihood ratio")).
                    withXaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.Y2),domain=(0, 1),range=(-0.05,0.6),title="False Positive Rate")).
                    withYaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.X2),domain=(0, 0.48),range=(0,1.05),title="True Positive Rate"))              

            Plotly.plot(s"$plotlyRootP/$fName-histVSpav.html", traces, layout)
        }

        def plotLLR_ROC_4Panes(llrLeft: Row,
                    llrRight: Row,        
                    fprLeft: Row,
                    tprLeft: Row,
                    fprRight: Row,
                    tprRight: Row,                    
                    threshLeft: Row,
                    threshRight: Row,
                    minusθLeft: Segment,
                    minusθRight: Segment,
                    isocostLeft: Segment,
                    isocostRight: Segment,
                    annotations: Option[Seq[Annotation]],
                    fName: String
        ) = {
            val llrLeftTrace = Scatter(threshLeft, llrLeft).
                withName("LLR (High AUC Recognizer)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X1).
                withYaxis(AxisReference.Y1).
                withLine(Line(color=Color.RGBA(162, 155, 155, 0.9), width = 2.5))
                
            val llrRightTrace = Scatter(threshRight, llrRight).
                withName("LLR (Low AUC Recognizer)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X3).
                withYaxis(AxisReference.Y3).
                withLine(Line(color=Color.RGBA(94, 30, 30, 0.9), width = 2.5))

            val rocLeftTrace = Scatter(fprLeft, tprLeft).
                withName("ROC (High AUC Recognizer)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X2).
                withYaxis(AxisReference.Y2).
                withLine(Line(color=Color.RGBA(162, 155, 155, 0.9), width = 2.5))

            val rocRightTrace = Scatter(fprRight, tprRight).
                withName("ROC (Low AUC Recognizer)").
                withMode(ScatterMode(ScatterMode.Lines)).
                withXaxis(AxisReference.X4).
                withYaxis(AxisReference.Y4).
                withLine(Line(color=Color.RGBA(94, 30, 30, 0.9), width = 2.5))

            val traces = Seq(llrLeftTrace,llrRightTrace,rocLeftTrace,rocRightTrace)

            // Add lines
            val thetaLeft = minusθLeft match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x1","y1")}
            val thetaRight = minusθRight match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x3","y3")}
            val rocLeft = isocostLeft match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x2","y2")}
            val rocRight = isocostRight match {case Segment(Point(x0,y0), Point(x1,y1)) => lineShape(x0,y0,x1,y1,"x4","y4")}
            
            val shapes = Seq(thetaLeft,thetaRight,rocLeft,rocRight)

            val layout =  Layout().
                    withTitle("High AUC (Left) vs Low AUC (Right)").
                    withWidth(1300).
                    withHeight(800).
                    withLegend(Legend().withX(0.05).withY(-0.3).withOrientation(Orientation.Horizontal).withYanchor(Anchor.Bottom).withXanchor(Anchor.Middle)). //(x=0.5,y=-0.3,xanchor=Anchor.Bottom,yanchor=Anchor.Bottom)).
                    withXaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.Y1),domain=(0, 0.48),range=(-3.5,+3.5),title="s")).
                    withYaxis(Axis(anchor=AxisAnchor.Reference(AxisReference.X1),domain=(0.55, 1),range=(-3.5,+3.5),title="Log-likelihood ratio")).
                    withXaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.Y2),domain=(0, 0.48),range=(-0.05,1.1),title="False Positive Rate")).
                    withYaxis2(Axis(anchor=AxisAnchor.Reference(AxisReference.X2),domain=(0, 0.45),range=(-0.05,1.1),title="True Positive Rate")).
                    withXaxis3(Axis(anchor=AxisAnchor.Reference(AxisReference.Y3),domain=(0.52, 1),range=(-3.5,+3.5),title="s")).
                    withYaxis3(Axis(anchor=AxisAnchor.Reference(AxisReference.X3),domain=(0.55, 1),range=(-3.5,+3.5),title="Log-likelihood ratio")).
                    withXaxis4(Axis(anchor=AxisAnchor.Reference(AxisReference.Y4),domain=(0.52, 1),range=(-0.05,1.1),title="False Positive Rate")).
                    withYaxis4(Axis(anchor=AxisAnchor.Reference(AxisReference.X4),domain=(0, 0.45),range=(-0.05,1.1),title="True Positive Rate")).                                        
                    withShapes(shapes).
                    withAnnotations(annotations)

            Plotly.plot(s"$plotlyRootP/$fName-llrRoc4panes.html", traces, layout)
        }
    }

    object Evals{
        case class Point(x: Double,y: Double)
        case class Segment(p1: Point, p2: Point)

        case class Tradeoff(w1Counts: Vector[Double], w0Counts: Vector[Double], thresholds: Vector[Double]) {
            val N = w1Counts.size
            require(w0Counts.size == N)
            require(thresholds.size == N)

            /* Various ways to present the score distributions */

            val asCCD: Matrix = {
                val w0pdf = pdf(w0Counts)
                val w1pdf = pdf(w1Counts)
                Vector(w0pdf,w1pdf)
            }

            val asLLR: Row = {
                val infLLR = logodds((pdf(w0Counts),pdf(w1Counts)))
                clipToFinite(infLLR)
            }

            val asPmissPfa: Matrix = {
                val pMiss = cdf(w1Counts) // lhsArea of p(x|w1)
                val pFa = rhsArea(w0Counts) // Same as fpr but in the asc order of scores
                Vector(pMiss,pFa)
            }

            val asROC: Matrix = {
                val fpr = (rhsArea andThen decreasing)(w0Counts)
                val tpr = (rhsArea andThen decreasing)(w1Counts)
                Vector(fpr,tpr)
            }

            def asDET: Matrix = ???
            
            /* Given application parameters, return optimal threshold and the corresponding expected risk */

            /* Find score cut-off point that minimises Risk by finding
            where LLRs intersect -θ.
            Return the corresponding index in the score Vector.
            */
            def minusθ(pa: AppParameters) = -1*paramToTheta(pa)

            def argminRisk(pa: AppParameters): Int = this.asLLR.getClosestIndex(minusθ(pa))

            def expectedRisks(pa: AppParameters): Row = {
                val risk: Tuple2[Double,Double] => Double = paramToRisk(pa)
                this.asPmissPfa.transpose.map{case Vector(pMiss,pFa) => risk((pMiss,pFa))}
            }

            def minS(pa: AppParameters): Double = {
                val ii = argminRisk(pa)
                thresholds(ii)
            }

            def minRisk(pa: AppParameters): Double = {
                val ii = argminRisk(pa)
                val bestPmissPfa = (this.asPmissPfa.apply(0)(ii),this.asPmissPfa.apply(1)(ii))
                paramToRisk(pa)(bestPmissPfa)
                //== expectedRisks(pa)(ii)
            }

            def ber(p_w1: Double): Double = minRisk(AppParameters(p_w1,1,1))

            def affine(y1: Double, x1: Double, x2: Double, slope: Double) = y1 + (x2-x1)*slope

            def isocost(pa: AppParameters): Segment = {
                val slope = exp(-1*paramToTheta(pa))
                val ii = argminRisk(pa)
                val roc = this.asROC
                val (bfpr, btpr) = (roc(0).reverse(ii), roc(1).reverse(ii)) // .reverse because roc curves are score inverted
                val (x1,x2) = (roc(0)(0), roc(0).last)
                val (y1, y2) = (affine(btpr,bfpr,x1,slope),
                                affine(btpr,bfpr,x2,slope)
                )
                Segment(Point(x1,y1),Point(x2,y2))
            }
        }    
    }

    // TODO: ROC for AUC comparison

    // Data examples and analyses
    object Part1{
        import Plots._, Data._, FitSystems._, Evals._
        object Demo11{
            def run = plotCCD(hisTo.asCCD(0),hisTo.asCCD(1),hisTo.thresholds,"demo11")
        }
        object Demo12{
            // In code
        }
        object Demo13{
            val vlines = Some(Seq(Segment(Point(hisTo.minRisk(pa),0),Point(hisTo.minRisk(pa),5))))
            val interval = Some(Segment(Point(simRisk.percentile(5),0.0),Point(simRisk.percentile(95),5)))
    
            def run = plotUnivarHist(simRisk,"SVM Expected vs Actual risk","Risk",vlines,interval,"demo13")
        }

        /* CCD,LLR and E(r) to illustrate that Bayes Decisions
            depdent on the application parameters.
        */
        object Demo14{
            val cutLine1 = Segment(Point(hisTo.minS(pa),0),Point(hisTo.minS(pa),0.2))
            val cutLine2 = Segment(Point(hisTo.minS(pa),-4),Point(hisTo.minS(pa),4))
            val cutLine3 = Segment(Point(hisTo.minS(pa),-4),Point(hisTo.minS(pa),4))
            val thetaLine = Segment(Point(-5,-1*paramToTheta(pa)),Point(+5,-1*paramToTheta(pa)))
            val minRiskLine = Segment(Point(-5,hisTo.minRisk(pa)),Point(+5,hisTo.minRisk(pa)))
            
            def run = plotCCD_LLR_E_r(hisTo.asCCD(0),
                        hisTo.asCCD(1),
                        hisTo.asLLR,
                        hisTo.expectedRisks(pa),
                        hisTo.thresholds,
                        cutLine1,
                        cutLine2,
                        cutLine3,
                        thetaLine,
                        minRiskLine,
                        "demo14"
            )
        }
        /* CCD,LLR and ROC to illustrate isocosts
        */
        object Demo15{
            val cutLine: Segment = Segment(Point(hisTo.minS(pa),0),Point(hisTo.minS(pa),0.2))
            val thetaLine = Segment(Point(-5,-1*paramToTheta(pa)),Point(+5,-1*paramToTheta(pa)))

            def run = plotCCD_LLR_ROC(hisTo.asCCD(0),
                        hisTo.asCCD(1),
                        hisTo.asLLR,
                        hisTo.asROC(0),
                        hisTo.asROC(1),
                        hisTo.thresholds,
                        cutLine,
                        thetaLine,
                        hisTo.isocost(pa),
                        "demo15"
            )
        }
        /* LLR and ROC for steppy and convex hull to illustrate
           the optimal operating points and its relationship with monotonicity
        */
        object Demo16{
            def run = plotLLR_ROC(manybinsTo.asLLR,
                pavTo.asLLR,
                manybinsTo.asROC(0),
                manybinsTo.asROC(1),
                pavTo.asROC(0),
                pavTo.asROC(1),
                manybinsTo.thresholds,
                pavTo.thresholds,
                "demo16"
            )
        }

        object Demo17{
            
            def run: Unit = {
                // P(hiAuc > lowAuc)
                /*
                val p = simAUC.pr(_ > 0)
                val α = 0.05
                println(p)
                assert(p > (1-α))
                */
                val minusθ = Segment(Point(-5,hiTo.minusθ(aucPa)),Point(+5,hiTo.minusθ(aucPa)))

                val commentary = Some(Seq(annotate(0.8,0.6,2,2,f"AUC = $hiAuc%.2f"),
                                annotate(0.8,0.6,4,4,f"AUC = $lowAuc%.2f"),
                                annotate(0.8,0.5,2,2,f"Min Risk = ${hiTo.minRisk(aucPa)}%.2f"),
                                annotate(0.8,0.5,4,4,f"Min Risk = ${lowTo.minRisk(aucPa)}%.2f"),
                                annotate(-2,3,1,1,f"-θ = ${hiTo.minusθ(aucPa)}%.1f",true),
                                annotate(0.03,0.8,2,2,f"slope = -θ = ${lowTo.minusθ(aucPa)}%.1f",true),
                                annotate(0.03,0.8,4,4,f"slope = -θ = ${lowTo.minusθ(aucPa)}%.1f",true),
                ))

                plotLLR_ROC_4Panes(hiTo.asLLR,
                    lowTo.asLLR,
                    hiTo.asROC(0),
                    hiTo.asROC(1),
                    lowTo.asROC(0),
                    lowTo.asROC(1),
                    hiTo.thresholds,
                    lowTo.thresholds,
                    minusθ,
                    minusθ,
                    hiTo.isocost(aucPa),
                    lowTo.isocost(aucPa),
                    commentary,
                    "Demo17"
                )
            }
        }
        
        // Get train data, assuming balanced labels
        val pa = AppParameters(p_w1=0.5,Cmiss=25,Cfa=5)
        val trainData: Seq[Transaction] = transact(pa.p_w1).sample(1_000)
        val trainDF = trainData.toArray.asDataFrame(trainSchema, rootP)

        // Fit an SVM recognizer
        val baseModel: Recognizer = SupportVectorMachine("svm", None)//RF("rf", None) 
        val recognizer = getRecognizer(baseModel, trainDF)

        // Transform predictions (logit transform) to have scores on ]-inf, +inf[
        // Evaluate CCD on Eval (no dev used) -- Bonus: kernel density
        // Demo11
        val loEval = xEval.map(recognizer).map(logit).toVector
        val tarPreds = loEval zip yEval filter{case (lo,y) => y == 1} map {_._1} 
        val nonPreds = loEval zip yEval filter{case (lo,y) => y == 0} map {_._1}   

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
        */
        def sample(data: Row, perc: Double) = {
            require(0 < perc && perc < 1)
            val mask = Distribution.bernoulli(perc).sample(data.size)
            data zip(mask) filter{case (v,m) => m == 1} map(_._1)
        }
        
        val (tarTiny, nonTiny) = (sample(tarPreds,0.02), sample(nonPreds,0.02))
        
        assert(naiveA(nonTiny,tarTiny) == smartA(nonTiny,tarTiny)) // true
        
        println(smartA(nonPreds,tarPreds))

        // TODO: Check that U is to test that A = 0.5
        // If so then it's practically useless?
        // check if I can use the estimated variance to compare A1 and A2 for 2 models

        /* Bayes Decision Rules
        Plot CCD,LLR and ROC for histogram bins with 2000 points
        - 20 bins for a smooth approximation 
        - Then 100 bins to have a concavity, a segway into PAV
        - PAV counts with 20 bins overlayed with 100 bins histogram, LLR and ROC with isocosts to show the link with risk-based decisions
        */

        // Convert score vectors to histogram counts
        val numBins = 30
        val min=loEval.min
        val max=loEval.max
        val w0HistCounts = histogram(nonPreds,numBins,min,max).map(_._2).toVector
        val w1HistCounts = histogram(tarPreds,numBins,min,max).map(_._2).toVector
        val histThresh = histogram(tarPreds,numBins,min,max).map(_._1.toDouble).toVector
        val hisTo = Tradeoff(w1HistCounts,w0HistCounts,histThresh)

        // TODO: apply the classifier approach to the estimators
        // Min(E(r)) validation
        val cutOff = hisTo.minS(pa)
        val thresholder: (Double => User) = score => if (score > cutOff) {Fraudster} else {Regular}
        def classifier:(Array[Double] => User) = recognizer andThen logit andThen thresholder

        def simulateTransact: Distribution[Double] = for {
            transaction <- transact(pa.p_w1)
            prediction = classifier(transaction.features.toArray)
            risk = cost(pa, transaction.UserType, prediction)
        } yield risk

        def simulateDataset: Distribution[Double] = simulateTransact.repeat(1000).map(_.sum / 1000.0)
        val simRisk: Row = simulateDataset.sample(500).toVector


        val nBins = 400
        val w0ManybinsCnts = histogram(nonPreds,nBins,min,max).map(_._2).toVector
        val w1ManybinsCnts = histogram(tarPreds,nBins,min,max).map(_._2).toVector
        val manybinsThresh = histogram(tarPreds,nBins,min,max).map(_._1.toDouble).toVector
        val manybinsTo = Tradeoff(w1ManybinsCnts,w0ManybinsCnts,manybinsThresh)

        // Fit pav on Eval and plot LLR (line chart)
        // Demo12
        val pav = new PAV(loEval, yEval, plodds)
        // TODO: REMOVE THIS METHOD
        //val (scores, llr) = pav.scoreVSlogodds // scores <-> pavLoEval 

        // PAV
        val w0PavCounts = pav.nonTars
        val w1PavCounts = pav.targets
        val pavThresh: Row = pav.pavFit.bins.map(_.getX)
        val pavTo = Tradeoff(w1PavCounts,w0PavCounts,pavThresh)

        val PP = Matrix(Row(pa.p_w1,1-pa.p_w1))
        //assert(pavDist.minRisk(pa)==expectedRisks(PP, pavDist.asPmissPfa, pa.Cfa, pa.Cmiss)) // compare with a manual computation of all risks using PP @ PmissPfa

        // AUC and Risk

        def splitScores(data: List[Score]): Tuple2[Row,Row] = {
            val tarS = data.filter(_.label==1).map(_.s).toVector
            val nonS = data.filter(_.label==0).map(_.s).toVector
            (nonS,tarS)            
        }

        def score2Auc(data: List[Score]): Double = {
            val (nonS,tarS) = splitScores(data)
            smartA(nonS,tarS)            
        }
        def hiAUCdata: Distribution[List[Score]] = HighAUC.normalLLR.repeat(1000)
        def lowAUCdata: Distribution[List[Score]] = LowAUC.normalLLR.repeat(1000)

        def simAUC: Distribution[Double] = for {
            h <- hiAUCdata
            l <- lowAUCdata
            hAuc = score2Auc(h)
            lAuc = score2Auc(l)
            diff = (hAuc - lAuc)
        } yield diff

        val hiEval: List[Score] = hiAUCdata.sample(1)(0)
        val lowEval: List[Score] = lowAUCdata.sample(1)(0)

        val hiAuc = score2Auc(hiEval)
        val lowAuc = score2Auc(lowEval)

        val hiSplit: List[Tuple2[Double,Int]] = for (obs <- hiEval) yield (obs.s, obs.label)
        val (hiScores,hiLabels) = hiSplit.toVector.unzip
        val lowSplit: List[Tuple2[Double,Int]] = for (obs <- lowEval) yield (obs.s, obs.label)
        val (lowScores,lowLabels) = lowSplit.toVector.unzip

        val hiPav = new PAV(hiScores, hiLabels, plodds)
        val lowPav = new PAV(lowScores, lowLabels, plodds)

        val w0HiCnts = hiPav.nonTars
        val w1HiCnts = hiPav.targets
        val hiThresh = hiPav.pavFit.bins.map(_.getX)
        
        val w0LowCnts = lowPav.nonTars
        val w1LowCnts = lowPav.targets
        val lowThresh = lowPav.pavFit.bins.map(_.getX)        

        val aucPa = AppParameters(0.5,5,80)
        val aucTheta = paramToTheta(aucPa)

        val hiTo = Tradeoff(w1HiCnts,w0HiCnts,hiThresh)
        val lowTo = Tradeoff(w1LowCnts,w0LowCnts,lowThresh)

        println(hiTo.minRisk(aucPa))
        println(lowTo.minRisk(aucPa))

        var minn = hiEval.map(_.s).min
        var maxx = hiEval.map(_.s).max
        val w0HiCntsHist = histogram(hiEval.filter(_.label==0).map(_.s).toVector,30,minn,maxx).map(_._2).toVector
        val w1HiCntsHist = histogram(hiEval.filter(_.label==1).map(_.s).toVector,30,minn,maxx).map(_._2).toVector
        val hiHistThresh =   histogram(hiEval.filter(_.label==0).map(_.s).toVector,30,minn,maxx).map(_._1.toDouble).toVector
        val hiHisTo = Tradeoff(w1HiCntsHist,w0HiCntsHist,hiHistThresh)

        minn = lowEval.map(_.s).min
        maxx = lowEval.map(_.s).max
        val w0LowCntsHist = histogram(lowEval.filter(_.label==0).map(_.s).toVector,30,minn,maxx).map(_._2).toVector
        val w1LowCntsHist = histogram(lowEval.filter(_.label==1).map(_.s).toVector,30,minn,maxx).map(_._2).toVector
        val lowHistThresh =  histogram(lowEval.filter(_.label==0).map(_.s).toVector,30,minn,maxx).map(_._1.toDouble).toVector
        val lowHisTo = Tradeoff(w1LowCntsHist,w0LowCntsHist,lowHistThresh)
    }
}



object Entry{
    import Recipes._, Part1._
    
    def main(args: Array[String]): Unit = {
        //Demo11.run
        //Demo13.run
        //Demo14.run
        //Demo15.run
        //Demo16.run
        Demo17.run
  }
}
