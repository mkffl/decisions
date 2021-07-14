package decisions

import smile.math.MathEx.{log}
import java.io._

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
    }

    trait LinAlg{
        type Row = Vector[Double]
        type Matrix = Vector[Vector[Double]]

        def Row(xs: Int*) = Vector(xs: _*)
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

        def dotProduct(A: Matrix, B: Matrix) = {
            for ( (rowA, rowB) <- A zip B)
            yield rowA zip rowB map Function.tupled(_*_) reduceLeft (_+_)
        }        
    }
    trait MathHelp{
        def logit(x: Double): Double = {
            log(x/(1-x))
        }        
    }
}
