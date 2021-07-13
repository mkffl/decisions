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

    trait LinAlg
    trait MathHelp{
        def logit(x: Double): Double = {
            log(x/(1-x))
        }        
    }
}
