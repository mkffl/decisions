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

import decisions.Shared._, LinAlg._, Stats._, FileIO._, RowLinAlg._,
MatLinAlg._, CollectionsStats._
import decisions._, EvalUtils._, To._, Concordance._
import TransactionsData._, AUC._
import SmileKitLearn._, SmileFrame._
import Dataset._

/** Group all analyses and charts. */
object Recipes extends decisions.Systems {

  /** Instantiate data and related methods for the fraud use case. */
  object Data {
    val p_w1 = 0.3
    val trainData: Seq[Transaction] = transact(p_w1).sample(1_000)
    val devData: Seq[Transaction] = transact(p_w1).sample(100_000)
    val evalData: Seq[Transaction] = transact(p_w1).sample(5_000)
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
      new StructField("f10", DataTypes.DoubleType)
    )
    def extractFeatures(row: Transaction): Array[Double] = row match {
      case Transaction(u, feats) => feats.toArray
    }
    def extractUser(row: Transaction): Int = row match {
      case Transaction(Regular, feats)   => 0
      case Transaction(Fraudster, feats) => 1
    }

    val trainDF = trainData.toArray.asDataFrame(trainSchema, rootP)
    val xDev = devData.map(extractFeatures).toArray
    val xEval = evalData.map(extractFeatures).toArray
    val yDev = devData.map(extractUser).toArray
    val yEval = evalData.map(extractUser).toVector
    val plodds: Vector[Double] =
      (BigDecimal(-5.0) to BigDecimal(5.0) by 0.25).map(_.toDouble).toVector
  }

  /** Helper methods to fit recognizers and calibrators. */
  object FitSystems {
    val formula = Formula.lhs("label")

    def getRecognizer(
        model: Recognizer,
        trainDF: DataFrame
    ): (Array[Double] => Double) = model match {
      case m: Logit =>
        LogisticRegression
          .fit(formula, trainDF)
          .predictProba //TODO: add properties
      case m: RF =>
        RandomForest
          .fit(formula, trainDF, m.params.getOrElse(new Properties()))
          .predictProba
      case m: SupportVectorMachine => { //TODO: add properties
        val X = formula.x(trainDF).toArray
        val y = formula.y(trainDF).toIntArray.map { case 0 => -1; case 1 => 1 }
        val kernel = new GaussianKernel(8.0)
        SVM.fit(X, y, kernel, 5, 1e-3).predictProba
      }
    }

    def getCalibrator(
        model: Calibrator,
        pDev: Array[Double],
        yDev: Array[Int]
    ): (Double => Double) = model match {
      case Uncalibrated => x => x
      case c: Isotonic =>
        IsotonicRegressionScaling.fit(pDev, yDev).predict //.predictProba
    }
    def getName(tr: Transformer) = tr match {
      case Logit(name, p)                => name
      case RF(name, p)                   => name
      case SupportVectorMachine(name, p) => name
      case Isotonic(name)                => name
      case Platt(name)                   => name
      case Uncalibrated                  => "Uncalibrated"
      case _                             => "Not referenced"
    }
  }

  /* Helper methods for DCF simulation
   */
  object Simulations {

    /** Normalising constant used to go from dcf to ber */
    def getConstant(pa: AppParameters): Double =
      pa.p_w1 * pa.Cmiss + (1 - pa.p_w1) * pa.Cfa

    /** Expected risk simulation
      *
      * @param nRows the number of rows in the simulated dataset
      * @param pa the application type
      * @param data the transaction's data generation process
      * @param classifier a predictive pipeline that outputs the user type
      */
    def oneClassifierExpectedRisk(
        nRows: Integer,
        pa: AppParameters,
        data: Distribution[Transaction],
        classifier: (Array[Double] => User)
    ): Distribution[Double] = data
      .map { transaction =>
        {
          val binaryPrediction = classifier(
            transaction.features.toArray
          ) // Generate a transaction's predicted user and
          val dcf = cost(
            pa,
            transaction.UserType,
            binaryPrediction
          ) // calculate its dcf
          dcf
        }
      }
      .repeat(nRows) // Generate a dataset of dcf's
      .map { values =>
        values.sum.toDouble / nRows // Get the average dcf
      }

    /** Error rate simulations
      *
      * @param nRows the number of rows in the simulated dataset
      * @param pa the application type
      * @param data the transaction's data generation process
      * @param system1 a predictive pipeline that outputs the user type
      * @param system2 the alternative predictive pipeline
      */
    def twoSystemErrorRates(
        nRows: Integer,
        pa: AppParameters,
        data: Distribution[Transaction],
        system1: (Array[Double] => User),
        system2: (Array[Double] => User)
    ): Distribution[(Double, Double)] = data
      .map { transaction =>
        {
          val binaryPrediction1 = system1(
            transaction.features.toArray
          ) // Generate a transaction's predicted user and
          val dcf1 = cost(
            pa,
            transaction.UserType,
            binaryPrediction1
          ) // calculate its dcf
          val binaryPrediction2 =
            system2(transaction.features.toArray) // Same with system2
          val dcf2 = cost(pa, transaction.UserType, binaryPrediction2)
          (dcf1, dcf2)
        }
      }
      .repeat(nRows)
      . // Generate a dataset of dcf's
      map { listOfTup =>
        listOfTup unzip match { // Get the sum of dcf's
          case (l1, l2) => (l1.sum, l2.sum)
        }
      }
      .map { case (sum1, sum2) =>
        (sum1 / nRows, sum2 / nRows) // Get the average dcf
      }
      .map { case (avg1, avg2) =>
        (
          avg1 / getConstant(pa),
          (avg2 / getConstant(pa))
        ) // Convert to a Bayes error rate
      }

  }

  /** Plotly methods used by Demo{xy} objects to generate charts. */
  object Plots {

    /** Construct an Annotation for up to 4 chart locations. */
    def annotate(
        x: Double,
        y: Double,
        xref: Int,
        yref: Int,
        text: String,
        showarrow: Boolean = false
    ): Annotation = new Annotation()
      .withXref(xref match {
        case 1 => Ref.Axis(AxisReference.X1)
        case 2 => Ref.Axis(AxisReference.X2)
        case 3 => Ref.Axis(AxisReference.X3)
        case 4 => Ref.Axis(AxisReference.X4)
      })
      .withYref(xref match {
        case 1 => Ref.Axis(AxisReference.Y1)
        case 2 => Ref.Axis(AxisReference.Y2)
        case 3 => Ref.Axis(AxisReference.Y3)
        case 4 => Ref.Axis(AxisReference.Y4)
      })
      .withX(x)
      .withY(y)
      .withText(text)
      .withShowarrow(showarrow)

    /** Construct a Line with the default style arguments used for the blog charts. */
    def lineShape(
        x0: Double,
        y0: Double,
        x1: Double,
        y1: Double,
        xref: String = "x",
        yref: String = "y",
        c: Seq[Int] = Seq(129, 2, 2, 1)
    ): Shape = new Shape(
      `type` = Some("line"),
      xref = Some(xref),
      yref = Some(yref),
      x0 = Some(s"$x0"),
      y0 = Some(y0),
      x1 = Some(s"$x1"),
      y1 = Some(y1),
      fillcolor = None,
      opacity = None,
      line = Some(
        Line(
          color = Color.RGBA(c(0), c(1), c(2), c(3)),
          width = 1.0,
          dash = Dash.Dot
        )
      )
    )

    /** Construct a Rectange shape with the default style arguments used for the blog charts. */
    def rectangleShape(
        x0: Double,
        y0: Double,
        x1: Double,
        y1: Double,
        xref: String = "x",
        yref: String = "y",
        c: Seq[Int] = Seq(156, 165, 196, 1)
    ): Shape = new Shape(
      `type` = Some("rect"),
      xref = Some("x"),
      yref = Some("y"),
      x0 = Some(s"$x0"),
      y0 = Some(y0),
      x1 = Some(s"$x1"),
      y1 = Some(y1),
      fillcolor = Some(Color.RGBA(c(0), c(1), c(2), c(3))),
      opacity = Some(0.3),
      line = Some(Line(color = Color.RGBA(c(0), c(1), c(2), c(3)), width = 1.0))
    )

    val colours = List(
      Color.RGBA(162, 155, 155, 0.9),
      Color.RGBA(94, 30, 30, 0.9),
      Color.RGBA(170, 30, 200, 0.9),
      Color.RGBA(70, 80, 220, 0.9)
    )

    /** Class Conditional Distribution for one recogniser. */
    def plotCCD(
        w0pdf: Row,
        w1pdf: Row,
        thresholds: Row,
        vlines: Option[Seq[Segment]],
        fName: String,
        featName: String = "s",
        barWidth: Double = 0.14
    ) = {
      val ccdW0Trace = Bar(thresholds, w0pdf)
        .withName(s"p($featName|ω0)")
        .withMarker(
          Marker().withColor(Color.RGB(124, 135, 146)).withOpacity(0.5)
        )
        .withWidth(barWidth)

      val ccdW1Trace = Bar(thresholds, w1pdf)
        .withName(s"p($featName|ω1)")
        .withMarker(
          Marker().withColor(Color.RGB(6, 68, 91)).withOpacity(0.8)
        )
        .withWidth(barWidth)

      val traces = Seq(ccdW0Trace, ccdW1Trace)

      val lin: Option[Seq[Shape]] = vlines.map(xs =>
        xs.map { case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1)
        }
      )

      val layout = Layout()
        .withTitle("Class Conditional Distributions (CCD)")
        .withWidth(900)
        .withHeight(700)
        .withXaxis(Axis(title = featName, range = (-5, +5)))
        .withYaxis(Axis(title = s"p($featName|ω_i)"))
        .withShapes(lin)

      Plotly.plot(s"$plotlyRootP/$fName-ccd.html", traces, layout)
    }

    /** CCD, Log-likelihood ratio and E(risk) for one recogniser. */
    def plotCCD_LLR_E_r(
        w0pdf: Row,
        w1pdf: Row,
        llr: Row,
        e_r: Row,
        thresholds: Row,
        cutLine1: Segment,
        cutLine2: Segment,
        cutLine3: Segment,
        thetaLine: Segment,
        minRLine: Segment,
        annotations: Option[Seq[Annotation]],
        fName: String
    ) = {
      val ccdW0Trace = Bar(thresholds, w0pdf)
        .withName("P(s|ω0)")
        .withMarker(
          Marker().withColor(Color.RGB(124, 135, 146)).withOpacity(0.5)
        )
        .withWidth(0.5)
        .withXaxis(AxisReference.X1)
        .withYaxis(AxisReference.Y1)

      val ccdW1Trace = Bar(thresholds, w1pdf)
        .withName("P(s|ω1)")
        .withMarker(
          Marker().withColor(Color.RGB(6, 68, 91)).withOpacity(0.8)
        )
        .withWidth(0.5)
        .withXaxis(AxisReference.X1)
        .withYaxis(AxisReference.Y1)

      val llrTrace = Scatter(thresholds, llr)
        .withName("Log Likelihood Ratio")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X2)
        .withYaxis(AxisReference.Y2)

      val e_rTrace = Scatter(thresholds, e_r)
        .withName("Expected risk")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X3)
        .withYaxis(AxisReference.Y3)

      val traces = Seq(ccdW0Trace, ccdW1Trace, llrTrace, e_rTrace)

      val cutOff1 = cutLine1 match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x1", "y1")
      }
      val cutOff2 = cutLine2 match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x2", "y2")
      }
      val cutOff3 = cutLine3 match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x3", "y3")
      }
      val llrminTheta = thetaLine match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x2", "y2")
      }
      val minRisk = minRLine match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x3", "y3")
      }

      val shapes = Seq(cutOff1, cutOff2, cutOff3, llrminTheta, minRisk)

      val layout = Layout()
        .withTitle("Bayes Decisions")
        .withWidth(1000)
        .withHeight(900)
        .withXaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y1),
            domain = (0, 1),
            range = (-5, +5),
            title = "score (s)"
          )
        )
        .withYaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X1),
            domain = (0.65, 1),
            title = "P(s|ωi)"
          )
        )
        .withXaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y2),
            domain = (0, 1),
            range = (-5, +5),
            title = "score (s)"
          )
        )
        .withYaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X2),
            domain = (0.33, 0.65),
            title = "LLR(ω1)"
          )
        )
        .withXaxis3(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y3),
            domain = (0, 1),
            range = (-5, +5),
            title = "score (s)"
          )
        )
        .withYaxis3(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X3),
            domain = (0, 0.32),
            range = (-1, +5),
            title = "E(risk)"
          )
        )
        .withShapes(shapes)
        .withAnnotations(annotations)

      Plotly.plot(s"$plotlyRootP/$fName-bayesdecisions1.html", traces, layout)
    }

    /** Histogram for one dimensional data */
    def plotUnivarHist(
        observed: Row,
        title: String,
        xtitle: String,
        vlines: Option[Seq[Segment]],
        confidence: Option[Segment],
        annotations: Option[Seq[Annotation]],
        fName: String
    ) = {
      val conf: Option[Seq[Shape]] = confidence.map {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          Seq(rectangleShape(x0, y0, x1, y1))
      }

      val lin: Option[Seq[Shape]] = vlines.map(xs =>
        xs.map { case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1)
        }
      )

      val shapes: Option[Seq[Shape]] = for {
        c <- conf
        l <- lin
      } yield c ++ l

      val trace = Histogram(
        observed,
        name = title,
        histnorm = HistNorm.ProbabilityDensity
      )

      val layout = Layout()
        .withTitle(title)
        .withXaxis(Axis(title = xtitle))
        .withYaxis(Axis(title = "Frequency"))
        .withShapes(shapes)
        .withAnnotations(annotations)

      Plotly.plot(s"$plotlyRootP/$fName-simulation.html", Seq(trace), layout)
    }

    /** CCD, LLR and ROC curve for one recogniser. */
    def plotCCD_LLR_ROC(
        w0pdf: Row,
        w1pdf: Row,
        llr: Row,
        fpr: Row,
        tpr: Row,
        thresholds: Row,
        cutLine: Segment,
        thetaLine: Segment,
        rocLine: Segment,
        annotations: Option[Seq[Annotation]],
        fName: String
    ) = {
      val ccdW0Trace = Bar(thresholds, w0pdf)
        .withName("P(s|ω0)")
        .withMarker(
          Marker().withColor(Color.RGB(124, 135, 146)).withOpacity(0.5)
        )
        .withWidth(0.5)
        .withXaxis(AxisReference.X1)
        .withYaxis(AxisReference.Y1)

      val ccdW1Trace = Bar(thresholds, w1pdf)
        .withName("P(s|ω1)")
        .withMarker(
          Marker().withColor(Color.RGB(6, 68, 91)).withOpacity(0.8)
        )
        .withWidth(0.5)
        .withXaxis(AxisReference.X1)
        .withYaxis(AxisReference.Y1)

      val llrTrace = Scatter(thresholds, llr)
        .withName("Log Likelihood Ratio")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X2)
        .withYaxis(AxisReference.Y2)

      val rocTrace = Scatter(fpr, tpr)
        .withName("Receiving Operator Characteristics (ROC)")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X3)
        .withYaxis(AxisReference.Y3)

      val traces = Seq(ccdW0Trace, ccdW1Trace, llrTrace, rocTrace)

      val ccdCutOff = cutLine match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x1", "y1")
      }
      val llrminTheta = thetaLine match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x2", "y2")
      }
      val isocost = rocLine match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x3", "y3")
      }

      val shapes = Seq(ccdCutOff, llrminTheta, isocost)

      val layout = Layout()
        .withTitle("Bayes Decisions with the ROC")
        .withWidth(1000)
        .withHeight(900)
        .withXaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y1),
            domain = (0, 1),
            range = (-5, +5),
            title = "score (s)"
          )
        )
        .withYaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X1),
            domain = (0.65, 1),
            title = "P(s|ωi)"
          )
        )
        .withXaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y2),
            domain = (0, 1),
            range = (-5, +5),
            title = "score (s)"
          )
        )
        .withYaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X2),
            domain = (0.33, 0.65),
            title = "log-likelihood ratio"
          )
        )
        .withXaxis3(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y3),
            domain = (0, 1),
            range = (-0.05, 0.6),
            title = "False Positive Rate"
          )
        )
        .withYaxis3(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X3),
            domain = (0, 0.32),
            range = (0, 1.05),
            title = "True Positive Rate"
          )
        )
        .withAnnotations(annotations)
        .withShapes(shapes)

      Plotly.plot(s"$plotlyRootP/$fName-bayesdecisions2.html", traces, layout)
    }

    /** LLR and ROC curve for one recogniser. */
    def plotLLR_ROC(
        llr: Row,
        llrPAV: Row,
        fpr: Row,
        tpr: Row,
        fprPAV: Row,
        tprPAV: Row,
        thresholds: Row,
        thresholdsPAV: Row,
        fName: String
    ) = {
      val llrTrace = Scatter(thresholds, llr)
        .withName("LLR (histogram)")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X1)
        .withYaxis(AxisReference.Y1)

      val llrPAVTrace = Scatter(thresholdsPAV, llrPAV)
        .withName("LLR (PAV)")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X1)
        .withYaxis(AxisReference.Y1)

      val rocTrace = Scatter(fpr, tpr)
        .withName("ROC (histogram)")
        .withText(thresholds.map(_.toString))
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X2)
        .withYaxis(AxisReference.Y2)

      val rocPAVTrace = Scatter(fprPAV, tprPAV)
        .withName("ROC (PAV)")
        .withText(thresholdsPAV.map(_.toString))
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X2)
        .withYaxis(AxisReference.Y2)

      val traces = Seq(llrTrace, llrPAVTrace, rocTrace, rocPAVTrace)

      val layout = Layout()
        .withTitle("LLR and ROC curves - Histogram vs PAV")
        .withWidth(1000)
        .withHeight(900)
        .withXaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y1),
            domain = (0, 1),
            range = (-5, +5),
            title = "score (s)"
          )
        )
        .withYaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X1),
            domain = (0.52, 1),
            title = "Log-likelihood ratio"
          )
        )
        .withXaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y2),
            domain = (0, 1),
            range = (-0.05, 0.6),
            title = "False Positive Rate"
          )
        )
        .withYaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X2),
            domain = (0, 0.48),
            range = (0, 1.05),
            title = "True Positive Rate"
          )
        )

      Plotly.plot(s"$plotlyRootP/$fName-histVSpav.html", traces, layout)
    }

    /** Side-by-side LLR and ROC curves to compare 2 recognisers. */
    def plotLLR_ROC_4Panes(
        llrLeft: Row,
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
      val llrLeftTrace = Scatter(threshLeft, llrLeft)
        .withName("LLR (High AUC Recognizer)")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X1)
        .withYaxis(AxisReference.Y1)
        .withLine(Line(color = Color.RGBA(162, 155, 155, 0.9), width = 2.5))

      val llrRightTrace = Scatter(threshRight, llrRight)
        .withName("LLR (Low AUC Recognizer)")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X3)
        .withYaxis(AxisReference.Y3)
        .withLine(Line(color = Color.RGBA(94, 30, 30, 0.9), width = 2.5))

      val rocLeftTrace = Scatter(fprLeft, tprLeft)
        .withName("ROC (High AUC Recognizer)")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X2)
        .withYaxis(AxisReference.Y2)
        .withLine(Line(color = Color.RGBA(162, 155, 155, 0.9), width = 2.5))

      val rocRightTrace = Scatter(fprRight, tprRight)
        .withName("ROC (Low AUC Recognizer)")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withXaxis(AxisReference.X4)
        .withYaxis(AxisReference.Y4)
        .withLine(Line(color = Color.RGBA(94, 30, 30, 0.9), width = 2.5))

      val traces = Seq(llrLeftTrace, llrRightTrace, rocLeftTrace, rocRightTrace)

      // Add lines
      val thetaLeft = minusθLeft match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x1", "y1")
      }
      val thetaRight = minusθRight match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x3", "y3")
      }
      val rocLeft = isocostLeft match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x2", "y2")
      }
      val rocRight = isocostRight match {
        case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x4", "y4")
      }

      val shapes = Seq(thetaLeft, thetaRight, rocLeft, rocRight)

      val layout = Layout()
        .withTitle("High AUC (Left) vs Low AUC (Right)")
        .withWidth(1300)
        .withHeight(800)
        .withLegend(
          Legend()
            .withX(0.05)
            .withY(-0.3)
            .withOrientation(Orientation.Horizontal)
            .withYanchor(Anchor.Bottom)
            .withXanchor(Anchor.Middle)
        )
        .withXaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y1),
            domain = (0, 0.48),
            range = (-3.5, +3.5),
            title = "s"
          )
        )
        .withYaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X1),
            domain = (0.55, 1),
            range = (-3.5, +3.5),
            title = "Log-likelihood ratio"
          )
        )
        .withXaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y2),
            domain = (0, 0.48),
            range = (-0.05, 1.1),
            title = "False Positive Rate"
          )
        )
        .withYaxis2(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X2),
            domain = (0, 0.45),
            range = (-0.05, 1.1),
            title = "True Positive Rate"
          )
        )
        .withXaxis3(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y3),
            domain = (0.52, 1),
            range = (-3.5, +3.5),
            title = "s"
          )
        )
        .withYaxis3(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X3),
            domain = (0.55, 1),
            range = (-3.5, +3.5),
            title = "Log-likelihood ratio"
          )
        )
        .withXaxis4(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y4),
            domain = (0.52, 1),
            range = (-0.05, 1.1),
            title = "False Positive Rate"
          )
        )
        .withYaxis4(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X4),
            domain = (0, 0.45),
            range = (-0.05, 1.1),
            title = "True Positive Rate"
          )
        )
        .withShapes(shapes)
        .withAnnotations(annotations)

      Plotly.plot(s"$plotlyRootP/$fName-llrRoc4panes.html", traces, layout)
    }

    /** ROC curve for multiple overlaid recognisers.
      *  Can plot isocosts to compare recognisers and determine any outperformer.
      */
    def plotROC(
        fpr: Seq[Row],
        tpr: Seq[Row],
        thresholds: Seq[Row],
        titles: Seq[String],
        annotations: Option[Seq[Annotation]],
        lines: Option[Seq[Segment]],
        fName: String
    ) = {
      val traces = fpr.zip(tpr.zipWithIndex).map { case (x, (y, it)) =>
        Scatter(x, y)
          .withName(titles(it))
          .withMode(ScatterMode(ScatterMode.Lines))
          .withXaxis(AxisReference.X1)
          .withYaxis(AxisReference.Y1)
          .withLine(Line(color = colours.apply(it % colours.size), width = 2.5))
      }

      val shapes = lines.map(l =>
        l.map { case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1, "x1", "y1")
        }
      )

      val layout = Layout()
        .withTitle("ROC curves compared.")
        .withWidth(800)
        .withHeight(800)
        .withLegend(
          Legend()
            .withX(0.05)
            .withY(-0.3)
            .withYanchor(Anchor.Bottom)
            .withXanchor(Anchor.Middle)
        )
        .withXaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.Y1),
            range = (-0.1, 1.1),
            title = "False Positive Rate"
          )
        )
        .withYaxis(
          Axis(
            anchor = AxisAnchor.Reference(AxisReference.X1),
            range = (-0.1, 1.1),
            title = "True Positive Rate"
          )
        )
        .withAnnotations(annotations)
        .withShapes(shapes)

      Plotly.plot(s"$plotlyRootP/$fName-ROC-equal-utility.html", traces, layout)
    }

    /* Applied Probability of Error plot with all benchmarks
     * minimum DCF, EER and majority DCF
     */
    def plotAPE(
        recognizer: String,
        plo: Row,
        observedDCF: Row,
        minDCF: Row,
        eer: Double,
        majorityDCF: Row,
        fName: String
    ): Unit = {

      val observedDCFTrace = Scatter(plo, observedDCF)
        .withName("DCF")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(Line(color = Color.RGBA(94, 30, 30, 0.9), width = 2.5))

      val minDCFTrace = Scatter(plo, minDCF)
        .withName("Minimum DCF")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(Line(color = Color.RGBA(162, 155, 155, 0.9), width = 1.5))

      val EERTrace = Scatter(plo, plo.map(x => eer))
        .withName("EER")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(
          Line(
            color = Color.RGBA(162, 155, 155, 0.9),
            width = 1.5,
            dash = Dash.Dot
          )
        )

      val majorityTrace = Scatter(plo, majorityDCF)
        .withName("Majority Classifier DCF")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(Line(color = Color.RGBA(162, 155, 155, 0.9), width = 1.5))

      val layout = Layout()
        .withTitle("Applied Probability of Error (SVM + logit)")
        .withYaxis(Axis(range = (0.0, 0.2), title = "Error Probability"))
        .withXaxis(Axis(range = (-3.0, +3.0), title = "Application type (θ)"))
        .withWidth(800)
        .withHeight(800)

      val data = Seq(observedDCFTrace, minDCFTrace, EERTrace, majorityTrace)

      Plotly.plot(s"$plotlyRootP/$fName-ape.html", data, layout)

    }

    /* APE with two systems and EER as the only benchmark
     */
    def plotAPECompare(
        system1: String,
        system2: String,
        plo: Row,
        observedDCF1: Row,
        observedDCF2: Row,
        eer1: Double,
        eer2: Double,
        fName: String
    ): Unit = {
      val observedDCF1Trace = Scatter(plo, observedDCF1)
        .withName(system1)
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(Line(color = Color.RGBA(94, 30, 30, 0.9), width = 2.5))

      val observedDCF2Trace = Scatter(plo, observedDCF2)
        .withName(system2)
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(Line(color = Color.RGBA(6, 31, 156, 0.9), width = 2.5))

      val eer1Trace = Scatter(plo, plo.map(x => eer1))
        .withName("EER System 1")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(
          Line(
            color = Color.RGBA(94, 30, 30, 0.9),
            width = 1.5,
            dash = Dash.Dot
          )
        )

      val eer2Trace = Scatter(plo, plo.map(x => eer2))
        .withName("EER System 2")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(
          Line(
            color = Color.RGBA(6, 31, 156, 0.9),
            width = 1.5,
            dash = Dash.Dot
          )
        )

      val data = Seq(observedDCF1Trace, observedDCF2Trace, eer1Trace, eer2Trace)

      val layout = Layout()
        .withTitle(s"APE graphs for 2 systems")
        .withYaxis(Axis(range = (0.0, 0.2), title = "Error Probability"))
        .withXaxis(Axis(range = (-3.0, +3.0), title = "Application type (θ)"))
        .withWidth(800)
        .withHeight(800)

      Plotly.plot(s"$plotlyRootP/$fName-ape-compared.html", data, layout)
    }

    /** Class Conditional Distribution for one recogniser. */
    def plotSystemErrorRates(
        observations1: Row,
        observations2: Row,
        thresholds: Row,
        barWidth: Double = 0.14,
        vlines: Option[Seq[Segment]],
        //confidence: Option[Segment],
        //annotations: Option[Seq[Annotation]],
        fName: String
    ) = {
      val trace1 = Bar(thresholds, observations1)
        .withName("System 1")
        .withMarker(
          Marker().withColor(Color.RGB(124, 135, 146)).withOpacity(0.8)
        )
        .withWidth(barWidth)

      val trace2 = Bar(thresholds, observations2)
        .withName("System 2")
        .withMarker(
          Marker().withColor(Color.RGB(6, 68, 91)).withOpacity(0.8)
        )
        .withWidth(barWidth)

      val lin: Option[Seq[Shape]] = vlines.map(xs =>
        xs.map { case Segment(Point(x0, y0), Point(x1, y1)) =>
          lineShape(x0, y0, x1, y1)
        }
      )

      val layout = Layout()
        .withTitle("Error rate simulations - system 1 vs system 2")
        .withWidth(900)
        .withHeight(700)
        .withXaxis(Axis(title = "Error rate", range = (0.075, 0.105)))
        .withYaxis(Axis(title = s"Frequency", range = (0.0, 0.2)))
        .withShapes(lin)

      Plotly.plot(
        s"$plotlyRootP/$fName-2-histograms.html",
        Seq(trace1, trace2),
        layout
      )
    }

    /* Applied Probability of Error plot with all benchmarks
     * minimum DCF, EER and majority DCF
     */
    def plotReliabilityDiagram(
        accuracy1: Row,
        accuracy2: Row,
        frequency: Row,
        fName: String
    ): Unit = {

      val rd1Trace = Scatter(frequency, accuracy1)
        .withName("Recognizer 1")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(Line(color = Color.RGBA(94, 30, 30, 0.9), width = 2.5))

      val rd2Trace = Scatter(frequency, accuracy2)
        .withName("Recognizer 2")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(Line(color = Color.RGBA(162, 155, 155, 0.9), width = 1.5))

      val perfectCalibrationTrace = Scatter(frequency, frequency)
        .withName("Pefectly calibrated recognizer")
        .withMode(ScatterMode(ScatterMode.Lines))
        .withLine(
          Line(
            color = Color.RGBA(162, 155, 155, 0.9),
            width = 1.5,
            dash = Dash.Dot
          )
        )

      val layout = Layout()
        .withTitle("Applied Probability of Error (SVM + logit)")
        .withYaxis(Axis(range = (0.0, 1.0), title = "Accuracy"))
        .withXaxis(Axis(range = (0.0, 1.0), title = "Frequency"))
        .withWidth(1500)
        .withHeight(600)

      val data = Seq(rd1Trace, rd2Trace, perfectCalibrationTrace)

      Plotly.plot(s"$plotlyRootP/$fName-ReliabilityDiagram.html", data, layout)

    }

  }

  /** Data examples and analyses for parts 1 and 2 (although name suggests part 1 only).
    *  Data objects are often shared between different Demo objects, so the data is defined further down
    *  and available to all Demos.
    */
  object Part1 {
    import Plots._, Data._, FitSystems._, Simulations._
    object Demo11 {
      def run = plotCCD(
        manybinsTo.asCCD(0),
        manybinsTo.asCCD(1),
        manybinsTo.thresholds,
        None,
        "demo11"
      )
    }
    object Demo12 {
      val vlines = Some(
        Seq(
          Segment(Point(rfTo.minS(errorPa), 0), Point(rfTo.minS(errorPa), 0.15))
        )
      )

      def run = plotCCD(
        rfTo.asCCD(0),
        rfTo.asCCD(1),
        rfTo.thresholds,
        vlines,
        "demo12",
        "x",
        0.25
      )
    }
    object Demo13 {
      val E_r = hisTo.minRisk(pa)

      val vlines = Some(Seq(Segment(Point(E_r, 0), Point(E_r, 5))))
      val interval = Some(
        Segment(
          Point(simRisk.percentile(5), 0.0),
          Point(simRisk.percentile(95), 5)
        )
      )

      val commentary = Some(Seq(annotate(E_r, 2, 1, 1, f"E(r) = ${E_r}%.1f")))

      def run = plotUnivarHist(
        simRisk,
        "SVM Expected vs Actual risk",
        "Risk",
        vlines,
        interval,
        commentary,
        "demo13"
      )
    }

    /** CCD,LLR and E(r) to illustrate that Bayes Decisions depdend on the chosen application parameters. */
    object Demo14 {
      val c = hisTo.minS(pa)
      val minθ = minusθ(pa)
      val E_r = hisTo.minRisk(pa)

      val cutLine1 = Segment(Point(c, 0), Point(c, 0.2))
      val cutLine2 = Segment(Point(c, -4), Point(c, 4))
      val cutLine3 = Segment(Point(c, -4), Point(c, 4))
      val thetaLine = Segment(Point(-5, minusθ(pa)), Point(+5, minθ))
      val minRiskLine = Segment(Point(-5, E_r), Point(+5, E_r))

      val commentary = Some(
        Seq(
          annotate(c + 0.5, 0.15, 1, 1, f"c = ${c}%.1f"),
          annotate(2, minθ + 0.5, 2, 2, f"-θ = ${minθ}%.1f"),
          annotate(2, E_r + 0.2, 3, 3, f"min E(r) = ${E_r}%.1f")
        )
      )

      def run = plotCCD_LLR_E_r(
        hisTo.asCCD(0),
        hisTo.asCCD(1),
        hisTo.asLLR,
        hisTo.expectedRisks(pa),
        hisTo.thresholds,
        cutLine1,
        cutLine2,
        cutLine3,
        thetaLine,
        minRiskLine,
        commentary,
        "demo14"
      )
    }

    object Demo151 {
      /* lr(w1) =
                    p(s|w1)/p(s|w0)
       */
      def lr(to: Tradeoff) =
        pdf(to.w1Counts).zip(pdf(to.w0Counts)).map(tup => tup._1 / tup._2)

      /* slope =
                    pmiss(t)-pmiss(t-1) / pfa(t-1)-pfa(t)
       */
      def slope(to: Tradeoff) = {
        val pMissD = to
          .asPmissPfa(0)
          .sliding(2)
          .map { case Seq(x, y, _*) => y - x }
          .toVector
        val pFaD = to
          .asPmissPfa(1)
          .sliding(2)
          .map { case Seq(x, y, _*) => x - y }
          .toVector
        pMissD.zip(pFaD).map(tup => tup._1 / tup._2)
      }
    }

    /** CCD,LLR and ROC to illustrate isocosts. */
    object Demo152 {
      val cutLine: Segment =
        Segment(Point(hisTo.minS(pa), 0), Point(hisTo.minS(pa), 0.2))
      val thetaLine =
        Segment(Point(-5, -1 * paramToθ(pa)), Point(+5, -1 * paramToθ(pa)))

      val commentary = Some(
        Seq(
          annotate(-1, 0.15, 1, 1, f"s = ${hisTo.minS(pa)}%.2f"),
          annotate(-1.2, -2.0, 2, 2, f"-θ = ${minusθ(pa)}%.1f", false),
          annotate(
            0.1,
            0.8,
            3,
            3,
            f"slope = exp(-θ) = ${exp(minusθ(pa))}%.1f",
            false
          )
        )
      )

      def run = plotCCD_LLR_ROC(
        hisTo.asCCD(0),
        hisTo.asCCD(1),
        hisTo.asLLR,
        hisTo.asROC(0),
        hisTo.asROC(1),
        hisTo.thresholds,
        cutLine,
        thetaLine,
        hisTo.isocost(pa),
        commentary,
        "demo15"
      )
    }

    /** LLR and ROC for steppy and convex hull to illustrate
      * the optimal operating points and its relationship with monotonicity.
      */
    object Demo16 {
      def run = plotLLR_ROC(
        manybinsTo.asLLR,
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

    object Demo17 {

      def run: Unit = {
        // P(hiAuc > lowAuc) - uncomment to run (takes for ever on my small machine).
        /*
                val p = simAUC.pr(_ > 0)
                val α = 0.05
                println(p)
                assert(p > (1-α))
         */
        val minusTheta =
          Segment(Point(-5, minusθ(aucPa)), Point(+5, minusθ(aucPa)))

        val commentary = Some(
          Seq(
            annotate(0.8, 0.6, 2, 2, f"AUC = $hiAuc%.2f"),
            annotate(0.8, 0.6, 4, 4, f"AUC = $lowAuc%.2f"),
            annotate(0.8, 0.5, 2, 2, f"Min Risk = ${hiTo.minRisk(aucPa)}%.2f"),
            annotate(0.8, 0.5, 4, 4, f"Min Risk = ${lowTo.minRisk(aucPa)}%.2f"),
            annotate(-2, 3, 1, 1, f"-θ = ${minusθ(aucPa)}%.1f", true),
            annotate(
              0.03,
              0.8,
              2,
              2,
              f"slope = exp(-θ) = ${exp(minusθ(aucPa))}%.1f",
              true
            ),
            annotate(
              0.03,
              0.8,
              4,
              4,
              f"slope = exp(-θ) = ${exp(minusθ(aucPa))}%.1f",
              true
            )
          )
        )

        plotLLR_ROC_4Panes(
          hiTo.asLLR,
          lowTo.asLLR,
          hiTo.asROC(0),
          hiTo.asROC(1),
          lowTo.asROC(0),
          lowTo.asROC(1),
          hiTo.thresholds,
          lowTo.thresholds,
          minusTheta,
          minusTheta,
          hiTo.isocost(aucPa),
          lowTo.isocost(aucPa),
          commentary,
          "Demo17"
        )
      }
    }

    object Demo18 {
      def run: Unit = {
        val aucPaSteeper = AppParameters(0.05, 107, 190)

        val lines = Seq(
          majorityIsocost(aucPa),
          majorityIsocost(aucPaSteeper)
        )

        val fpr = Seq(hiTo.asROC, lowTo.asROC).map { case Vector(fpr, tpr) =>
          fpr
        }
        val tpr = Seq(hiTo.asROC, lowTo.asROC).map { case Vector(fpr, tpr) =>
          tpr
        }
        val thresholds = Seq(hiTo, lowTo).map {
          case Tradeoff(w1cnt, w0cnt, thresh) => thresh
        }
        val titles = Seq("High AUC model", "Low AUC model")

        val commentary = Some(
          Seq(
            annotate(
              0.8 / exp(minusθ(aucPa)),
              0.8,
              1,
              1,
              f"slope = ${exp(minusθ(aucPa))}%.1f",
              true
            ),
            annotate(
              1.0 / exp(minusθ(aucPaSteeper)),
              1.0,
              1,
              1,
              f"slope = ${exp(minusθ(aucPaSteeper))}%.1f",
              true
            )
          )
        )

        plotROC(
          fpr,
          tpr,
          thresholds,
          titles,
          commentary,
          Some(lines),
          "Demo18"
        )
      }
    }

    object Demo110 {
      def run: Unit = {
        val steppy = new SteppyCurve(loEval, yEval, plodds)
        val pav = new PAV(loEval, yEval, plodds)

        plotAPE(
          "svm",
          plodds,
          steppy.bayesErrorRate,
          pav.bayesErrorRate,
          pav.EER,
          steppy.majorityErrorRate,
          "Demo110"
        )
      }
    }

    object Demo111 {
      def run: Unit = {
        val steppy1 = new SteppyCurve(loEval, yEval, plodds)
        val pav1 = new PAV(loEval, yEval, plodds)
        val steppy2 = new SteppyCurve(altLoEval, yEval, plodds)
        val pav2 = new PAV(altLoEval, yEval, plodds)

        plotAPECompare(
          "SVM + logit",
          "Random Forest + logit",
          plodds,
          steppy1.bayesErrorRate,
          steppy2.bayesErrorRate,
          pav1.EER,
          pav2.EER,
          "Demo111"
        )
      }
    }

    object Demo112 {
      def getThresholder(cutOff: Double)(score: Double): User =
        if (score > cutOff) { Fraudster }
        else { Regular }

      val pa2 = AppParameters(0.3, 4.94, 1.0)
      def getConstant(pa: AppParameters) =
        pa.p_w1 * pa.Cmiss + (1 - pa.p_w1) * pa.Cfa
      val cst = getConstant(pa2)

      val targetTheta = 0.75
      val ii = plodds.getClosestIndex(targetTheta)
      val steppy1 = new SteppyCurve(loEval, yEval, plodds)
      val E_r1 = steppy1.bayesErrorRate(ii)
      val cutOff: Double = minusθ(pa2)
      def bayesThreshold1: Double => User = getThresholder(cutOff) _
      def system1: Array[Double] => User =
        recognizer andThen logit andThen bayesThreshold1

      val steppy2 = new SteppyCurve(altLoEval, yEval, plodds)
      val E_r2 = steppy2.bayesErrorRate(ii)
      def bayesThreshold2 = getThresholder(cutOff) _
      def system2 = altRecognizer andThen logit andThen bayesThreshold2

      println(targetTheta)
      println(E_r1)
      println(E_r2)
      println(cutOff)
      println(cst)

      val nSamples = 300

      val (berSystem1, berSystem2) = twoSystemErrorRates(
        5000,
        pa2,
        transact(pa2.p_w1),
        system1,
        system2
      ).sample(nSamples).toVector.unzip

      println(berSystem1.zip(berSystem2))

      val binned1 = histogram(berSystem1, 15, berSystem1.min, berSystem1.max)
        .map(_._2)
        .map(_ / nSamples.toDouble)
        .toVector
      val binned2 = histogram(berSystem2, 15, berSystem2.min, berSystem2.max)
        .map(_._2)
        .map(_ / nSamples.toDouble)
        .toVector
      val thresholds = histogram(berSystem2, 15, berSystem2.min, berSystem2.max)
        .map(_._1.toDouble)
        .toVector

      val vlines = Some(
        Seq(
          Segment(Point(E_r1, 0), Point(E_r1, 1)),
          Segment(Point(E_r2, 0), Point(E_r2, 1))
        )
      )

      def run = plotSystemErrorRates(
        binned1,
        binned2,
        thresholds,
        0.002,
        vlines,
        "Demo112"
      )

      println(
        berSystem1
          .zip(berSystem2)
          .filter { case (b1, b2) => b1 < b2 }
          .size / nSamples.toDouble
      )
    }

    /** Create the data
      * Base model is an SVM, used throughout parts 1 and 2.
      * Alternative model is random forest - non-gaussian CCD make it visually interesting.
      */

    // Get train data, assuming balanced labels (p_w1=0.5)
    val pa = AppParameters(p_w1 = 0.5, Cmiss = 25, Cfa = 5)
    val errorPa = AppParameters(p_w1 = 0.5, 1, 1)
    val trainData: Seq[Transaction] = transact(pa.p_w1).sample(1_000)
    val trainDF = trainData.toArray.asDataFrame(trainSchema, rootP)

    // Fit recognizers
    // Standard recognizer is based on SVM
    val baseModel: Recognizer = SupportVectorMachine("svm", None)
    val recognizer = getRecognizer(baseModel, trainDF)

    // Alternative recognizer is based on random forests
    val altModel: Recognizer = RF("rf", None)
    val altRecognizer = getRecognizer(altModel, trainDF)

    // Transform predictions (logit transform) to have scores on ]-inf, +inf[
    // Evaluate CCD on Eval (no dev used)
    val loEval = xEval map (recognizer) map (logit) toVector
    val tarPreds = loEval zip yEval filter { case (lo, y) => y == 1 } map {
      _._1
    }
    val nonPreds = loEval zip yEval filter { case (lo, y) => y == 0 } map {
      _._1
    }

    val altLoEval = xEval map (altRecognizer) map (logit) toVector

    // Histograms
    // rfTo used in Demo12 to get skewed-shaped CCD
    // manybinsTo used in demo11 to get thin-sized bins, also used in Demo16 as a segway into PAV
    val hisTo = makeHisTo(loEval, yEval, numBins = 30)
    val rfTo = makeHisTo(altLoEval, yEval)
    val manybinsTo = makeHisTo(loEval, yEval, 400)

    // Simulation to validate Min(E(r)) (Demo13)
    val cutOff: Double = hisTo.minS(pa)
    val nrows = 1000
    val nsimulations = 500

    val thresholder: (Double => User) = score =>
      if (score > cutOff) { Fraudster }
      else { Regular }

    def classifier: (Array[Double] => User) =
      recognizer andThen logit andThen thresholder

    val simRisk: Row =
      oneClassifierExpectedRisk(1000, pa, transact(pa.p_w1), classifier)
        .sample(500)
        .toVector

    // Fit pav on Eval and plot LLR (Demo16)
    val pav = new PAV(loEval, yEval, plodds)
    val w0PavCounts = pav.nonTars
    val w1PavCounts = pav.targets
    val pavThresh: Row = pav.pavFit.bins.map(_.getX)
    val pavTo: Tradeoff = Tradeoff(w1PavCounts, w0PavCounts, pavThresh)

    // ************* Start AUC

    // AUC and Risk use case (Demo17 and Demo18)
    def splitScores(data: List[Score]): Tuple2[Row, Row] = {
      val tarS = data.filter(_.label == 1).map(_.s).toVector
      val nonS = data.filter(_.label == 0).map(_.s).toVector
      (nonS, tarS)
    }

    def score2Auc(data: List[Score]): Double = {
      val (nonS, tarS) = splitScores(data)
      smartA(nonS, tarS)
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

    /** Histogram from predictions
      *
      * @param eval the sequence of score predictins and ground truth
      * @return the Tradeoff object
      */
    def PAV2Hist(eval: List[Score]): Tradeoff = {
      val split: List[Tuple2[Double, Int]] =
        for (obs <- eval) yield (obs.s, obs.label)
      val (scores, labels) = split.toVector.unzip
      val pav = new PAV(scores, labels, plodds)

      val w0Cnts = pav.nonTars.map(clipTo1)
      val w1Cnts = pav.targets.map(clipTo1)
      val thresh = pav.pavFit.bins.map(_.getX)

      Tradeoff(w1Cnts, w0Cnts, thresh)
    }

    val aucPa = AppParameters(0.05, 107, 90)

    val hiEval: List[Score] = hiAUCdata.sample(1)(0)
    val lowEval: List[Score] = lowAUCdata.sample(1)(0)
    val hiAuc = score2Auc(hiEval)
    val lowAuc = score2Auc(lowEval)

    val hiTo = PAV2Hist(hiEval)
    val lowTo = PAV2Hist(lowEval)

    // ************* End AUC
  }
}

object Entry {
  import Recipes._, Part1._

  def main(args: Array[String]): Unit = {
    // Demo11.run
    // Demo12.run
    Demo13.run
    // Demo14.run
    // Demo152.run
    // Demo16.run
    // Demo17.run
    // Demo18.run
    // Demo110.run
    // Demo111.run
    // Demo112.run
  }
}
