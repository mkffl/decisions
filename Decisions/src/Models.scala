package decisions
import smile.classification._
import smile.data.{Tuple, DataFrame}
import smile.data.`type`._
import smile.io.Read

import org.apache.commons.csv.CSVFormat 

import decisions.CompareSystems.utils._

object Dataset{
    trait DecisionDataset
    trait RecognizerDataset extends DecisionDataset
    trait CalibrationDataset extends DecisionDataset
    trait EvaluationDataset extends DecisionDataset
}

object Systems{
    trait Transformer
    trait Recognizer extends Transformer
    trait Calibrator extends Transformer

    case class Logit(name: String) extends Recognizer
    case class RF(name: String) extends Recognizer
    case class Isotonic(name: String) extends Calibrator
    case class Platt(name: String) extends Calibrator
    case object Uncalibrated extends Calibrator

    type System = Tuple2[Recognizer, Calibrator]    
}



class SmileKitLearn[T](model: T){
    def predictproba(data: Array[Array[Double]]): Array[Double] = model match {
        case m:LogisticRegression => {
            val tempScore = new Array[Double](2) // 2 classes
            data.map{x => 
                m.predict(x, tempScore)
                tempScore(1)
            }
        }
        case m:RandomForest => {
            val schema = m.schema // model schema excludes the target variable...
            val dataTupled = for (row <- data) yield Tuple.of(row, schema)
            val tempScore = new Array[Double](2) // 2 classes
            dataTupled.map{x => 
                m.predict(x, tempScore)
                tempScore(1)
            }
        }
    }
}             

object SmileKitLearn {
    implicit def logitToLogit(m: LogisticRegression): SmileKitLearn[LogisticRegression] =
        new SmileKitLearn[LogisticRegression](m)
    implicit def rfToRf(m: RandomForest): SmileKitLearn[RandomForest] =
        new SmileKitLearn[RandomForest](m)
}



class SmileFrame{
    val format = CSVFormat.DEFAULT.withFirstRecordAsHeader()

    def asDataFrame(data: Seq[String], schema: StructType, dataType: Dataset.DecisionDataset, rootPath: String): DataFrame = dataType match {
        case t:Dataset.RecognizerDataset => {
            writeFile(s"$rootPath/train.csv", data, "label,amount,count")
            val train: DataFrame = Read.csv(s"$rootPath/train.csv", format, schema)
            train
        }
        case t:Dataset.CalibrationDataset => {
            writeFile(s"$rootPath/calibration.csv", data, "label,score")
            val calibration: DataFrame = Read.csv(s"$rootPath/calibration.csv", format, schema)
            calibration
        }
    }
}

object SmileFrame {
    implicit def ArrayToDataFrame(d: Array[Array[String]]): SmileFrame =
        new SmileFrame()
}