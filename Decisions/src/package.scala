package decisions

import smile.math.MathEx.{log}
import java.io._
import smile.math.MathEx.{logistic, min, log}
import decisions.TransactionsData._

package object Shared{
    trait FileIO {
        def writeFile(filename: String, lines: Seq[String], headers: String): Unit = {
            val file = new File(filename)
            val bw = new BufferedWriter(new FileWriter(file))
            val linesToPrint = Seq(headers) ++ lines
            for (line <- linesToPrint) {
                bw.write(s"$line\n")
            }
            bw.close()
        }

        val plotlyRootP = "/Users/michel/Documents/pjs/model-discrimination-calibration/Stats-Decisions/outputs"

        def stringifyVector(data: Vector[Double]): String = data.map(value => s"$value,").mkString.stripSuffix(",")
    }

    trait LinAlg{

        type Row = Vector[Double]
        type Matrix = Vector[Vector[Double]]

        def Row[T](xs: T*) = Vector(xs: _*)
        def Matrix(xs: Row*) = Vector(xs: _*)

        implicit def intToDoubleMat(mat: Vector[Vector[Int]]): Matrix = 
            for (row <- mat) 
                yield for (value <- row) 
                    yield value.toDouble
        
        implicit def intToDoubleRow(vect: Vector[Int]): Row = 
            for (value <- vect) yield value.toDouble
        
        implicit def floatToDoubleRow(row: Vector[Float]): Row = 
            for (value <- row)
                    yield value.toDouble
        
        class RowLinAlg(a: Row){
            /* Inner product
            */
            def dot(b: Row): Double =  a zip b map Function.tupled(_*_) reduceLeft (_+_)

            /* Common element-wise operations
            */
            def +(b: Row): Row = a zip b map Function.tupled(_+_)
            def -(b: Row): Row = a zip b map Function.tupled(_-_)
            def *(b: Row): Row = a zip b map Function.tupled(_*_)
        }
        
        object RowLinAlg {
            implicit def VectorToRow(a: Row): RowLinAlg =
                new RowLinAlg(a)
        }

        class MatLinAlg(A: Matrix){
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
            */
            def *(B: Matrix): Matrix = {
                for ((rowA, rowB) <- A zip B)
                yield rowA zip rowB map Function.tupled(_*_)
            }
        }
        
        object MatLinAlg {
            implicit def vecVecToMat(a: Matrix): MatLinAlg =
                new MatLinAlg(a)
        }
    }
    trait MathHelp{

        def logit(x: Double): Double = {
            log(x/(1-x))
        }

        def expit(x: Double)= logistic(x)
        }

        class CollectionsStats(c: Vector[Double]){
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
        }

        object CollectionsStats{
            implicit def toCollectionsStats(c: Vector[Double]): CollectionsStats =
                new CollectionsStats(c)
        }    

    trait Validation {
        case class AppParameters(p_w1: Double, Cmiss: Double, Cfa: Double)

        def cost(p: AppParameters, actual: User, pred: User): Double = pred match {
                case Fraudster if actual == Regular => p.Cfa
                case Regular if actual == Fraudster => p.Cmiss
                case _ => 0.0
        }

        def paramToTheta(p: AppParameters): Double = log(p.p_w1/(1-p.p_w1)*(p.Cmiss/p.Cfa))
    }
}
