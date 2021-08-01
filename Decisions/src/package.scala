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

        implicit def intToDouble(mat: Vector[Vector[Int]]): Matrix = 
            for (row <- mat) 
                yield for (value <- row) 
                    yield value.toDouble
        
        implicit def floatToDoubleRow(row: Vector[Float]): Row = 
            for (value <- row)
                    yield value.toDouble

        /* Matrix product operation
        If matrix A is (m,n) and B is (n,p) then output is (m,p)
        Input matrices must be (m,n) and (n,p).
        */
        def matMul(A: Matrix, B: Matrix) = {
            for (row <- A) // (m,n)
            yield for(col <- B.transpose) // rotate to (p,n) to loop thru the cols
                yield row zip col map Function.tupled(_*_) reduceLeft (_+_)
        }

        // Dot product sum?
        def dotProduct(A: Matrix, B: Matrix) = {
            for ( (rowA, rowB) <- A zip B)
            yield rowA zip rowB map Function.tupled(_*_) reduceLeft (_+_)
        }

        def dot(A: Matrix, B: Matrix): Matrix = {
            for {
                row <- A
            } yield for {
                col <- B.transpose
                paired = row zip col
                multipled = paired map Function.tupled(_*_)
                summed = multipled reduceLeft (_+_)
            } yield summed
        }

        def addVectors(A: Row, B: Row) = A zip B map Function.tupled(_+_)
    }
    trait MathHelp{
        def logit(x: Double): Double = {
            log(x/(1-x))
        }

        def expit(x: Double)= logistic(x)
    }

    trait Validation {
        case class AppParameters(p_w1: Double, Cmiss: Double, Cfa: Double)

        def cost(p: AppParameters, actual: User, pred: User): Double = pred match {
                case Fraudster if actual == Regular => p.Cfa
                case Regular if actual == Fraudster => p.Cmiss
                case _ => 0.0
        }
    }
}
