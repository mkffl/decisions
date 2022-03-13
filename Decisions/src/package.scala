package decisions

import smile.math.MathEx.{log}
import java.io._
import smile.math.MathEx.{logistic, min, log}
import decisions.TransactionsData._

package object Shared {
  type Row = Vector[Double]
  type Matrix = Vector[Vector[Double]]

  def Row[T](xs: T*) = Vector(xs: _*)
  def Matrix(xs: Row*) = Vector(xs: _*)
  object FileIO {
    def writeFile(
        filename: String,
        lines: Seq[String],
        headers: String
    ): Unit = {
      val file = new File(filename)
      val bw = new BufferedWriter(new FileWriter(file))
      val linesToPrint = Seq(headers) ++ lines
      for (line <- linesToPrint) {
        bw.write(s"$line\n")
      }
      bw.close()
    }

    val plotlyRootP =
      "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/temp"

    val rootP =
      "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs/transactions_data"

    def stringifyVector(data: Vector[Double]): String =
      data.map(value => s"$value,").mkString.stripSuffix(",")
  }

  object LinAlg {
    implicit def intToDoubleMat(mat: Vector[Vector[Int]]): Matrix =
      for (row <- mat)
        yield for (value <- row)
          yield value.toDouble

    implicit def intToDoubleRow(vect: Vector[Int]): Row =
      for (value <- vect) yield value.toDouble

    implicit def floatToDoubleRow(row: Vector[Float]): Row =
      for (value <- row)
        yield value.toDouble

    class RowLinAlg(a: Row) {
      /* Inner product
       */
      def dot(b: Row): Double =
        a zip b map Function.tupled(_ * _) reduceLeft (_ + _)

      /* Common element-wise operations
       */
      def +(b: Row): Row = a zip b map Function.tupled(_ + _)
      def -(b: Row): Row = a zip b map Function.tupled(_ - _)
      def *(b: Row): Row = a zip b map Function.tupled(_ * _)
    }

    object RowLinAlg {
      implicit def VectorToRow(a: Row): RowLinAlg =
        new RowLinAlg(a)
    }

    class MatLinAlg(A: Matrix) {
      import RowLinAlg._

      /* Dot multiplication
            Called "at" like in python 3.+
            (Symbol @ can't be used in scala)
       */
      def at(B: Matrix): Matrix = {
        for (row <- A)
          yield for {
            col <- B.transpose
          } yield row dot col
      }
      def dot(B: Matrix): Matrix = at(B)

      /* Element-wise operations
                If B has only one row then assume it needs to be broadcast
                to every row of A, as with numpy.
       */
      def *(B: Matrix): Matrix = B match {
        case Vector(uniqueRowB) =>
          for (rowA <- A) yield rowA zip uniqueRowB map Function.tupled(_ * _)
        case Vector(first, rest @ _*) =>
          for ((rowA, rowB) <- A zip B)
            yield rowA zip rowB map Function.tupled(_ * _)
        case _ => Matrix(Row(-10, -10))
      }
    }

    object MatLinAlg {
      implicit def vecVecToMat(a: Matrix): MatLinAlg =
        new MatLinAlg(a)
    }
  }
  object Stats {

    def logit(x: Double): Double = {
      log(x / (1 - x))
    }

    def expit(x: Double) = logistic(x)

    def exp(x: Double) = math.exp(x)

    /* Taken from probability_monad */
    def findBucketWidth(
        min: Double,
        max: Double,
        buckets: Int
    ): (BigDecimal, BigDecimal, BigDecimal, Int) = {
      // Use BigDecimal to avoid annoying rounding errors.
      val widths = List(0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0).map(
        BigDecimal.apply
      )
      val span = max - min
      val p = (math.log(span) / math.log(10)).toInt - 1
      val scale = BigDecimal(10).pow(p)
      val scaledWidths = widths.map(_ * scale)
      val bestWidth = scaledWidths.minBy(w => (span / w - buckets).abs)
      val outerMin = (min / bestWidth).toInt * bestWidth
      val outerMax = ((max / bestWidth).toInt + 1) * bestWidth
      val actualBuckets = ((outerMax - outerMin) / bestWidth).toInt
      (outerMin, outerMax, bestWidth, actualBuckets)
    }

    /* Taken from probability_monad */
    def histogram(data: Vector[Double], buckets: Int, min: Double, max: Double)(
        implicit
        ord: Ordering[Double],
        toDouble: Double <:< Double
    ): IndexedSeq[(scala.math.BigDecimal, Double)] = {
      //val min = data.head
      //val max = data.last
      val n = data.size
      val rm = BigDecimal.RoundingMode.DOWN

      val (outerMin, outerMax, unusedWidth, nbuckets) =
        findBucketWidth(toDouble(min), toDouble(max), buckets)
      val width = (outerMax - outerMin) / nbuckets
      def toBucket(a: Double): BigDecimal =
        ((toDouble(a) - outerMin) / width).setScale(0, rm) * width + outerMin
      val bucketToProb = data
        .groupBy(toBucket)
        .map({ case (b, vs) => b -> vs.size.toDouble })
        .toMap
      val bucketed = (outerMin to outerMax by width).map(a =>
        a -> bucketToProb.getOrElse(a, 0.0)
      )
      bucketed
    }

    class CollectionsStats(c: Vector[Double]) {
      def argmax: Int = c.zipWithIndex.maxBy(x => x._1)._2
      def argmin: Int = c.zipWithIndex.minBy(x => x._1)._2
      def mean: Double = c.sum / c.size.toDouble

      /* Percentile
                Returns v_p, the value in c such that p% values are
                inferior to v_p.

                Note: a more common approach is to interpolate the value of
                the two nearest neighbours in case the normalized ranking does not match
                the location of p exactly. If c is large, assumed here, then it won't
                make a big difference.
       */
      def percentile(p: Int) = {
        require(0 <= p && p <= 100)
        val sorted = c.sorted
        val ii = math.ceil((c.length - 1) * (p / 100.0)).toInt
        sorted(ii)
      }

      def median = percentile(50)

      /** Find index of closest number from the target in a list
        * {{
        * val target = 3.2
        * val nums = List(-2.0, 3.0, 4.0)
        * getClosestIndex(nums, target) // returns 1
        * }}
        */
      def getClosestIndex(target: Double): Integer =
        c.zipWithIndex.minBy(tup => math.abs(tup._1 - target))._2
    }

    object CollectionsStats {
      implicit def toCollectionsStats(c: Vector[Double]): CollectionsStats =
        new CollectionsStats(c)
    }

    // Define some of the common operations to estimate distributions
    def clipToFinite(data: Row, e: Double = 1e-6) = {
      val finite = data.filter(_.isFinite)
      val max = finite.max + e
      val min = finite.min - e
      data.map {
        case v if v.isNegInfinity => min
        case v if v.isPosInfinity => max
        case v                    => v
      }
    }

    val proportion: Row => Row = counts => {
      val S = counts.sum.toDouble
      counts.map(v => v / S)
    }
    val cumulative: Row => Row = freq => freq.scanLeft(0.0)(_ + _)
    val oneMinus: Row => Row = cdf => cdf.map(v => 1 - v)
    val decreasing: Row => Row = data => data.reverse
    val odds: Tuple2[Row, Row] => Row = w0w1 =>
      w0w1._1.zip(w0w1._2).map { case (non, tar) => tar / non }
    val logarithm: Row => Row = values => values.map(math.log)

    val pdf: Row => Row = proportion
    val cdf: Row => Row = pdf andThen cumulative
    val rhsArea: Row => Row = cdf andThen oneMinus
    val logodds: Tuple2[Row, Row] => Row = odds andThen logarithm

    /* Enforce floor counts of 1 to work around +/- inf values in LLR.

            Infinite likelihood-ratio values are particularly prevalent in thresholds near the edge, e.g. the counts of
            w0 near the max scores could be 0. This method replaces +/- inf LLR with small/big values, ensuring that they
            are considered for the optimal threshold (argminRisk).

     */
    def clipTo1(x: Int) = x match {
      case cnt if cnt == 0 => 1
      case cnt             => cnt
    }

  }
}
