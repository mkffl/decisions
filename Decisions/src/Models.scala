package decisions
import smile.classification._
import smile.data.{Tuple, DataFrame}
import smile.data.`type`._
import smile.io.Read

import org.apache.commons.csv.CSVFormat 

import decisions.CompareSystems.utils._
import decisions.TransactionsData._

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
    case class Bionomial(name: String) extends Calibrator
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
    def predictproba(data: Array[Double]): Array[Double] = model match {
        case m: IsotonicRegressionScaling => {
            data.map(m.predict)
        }
    }
    def predictProba(data: Array[Double]): Double = model match {
        case m:LogisticRegression => {
            val tempScore = new Array[Double](2) // 2 classes
            m.predict(data, tempScore)
            tempScore(1)
        }
        case m:RandomForest => {
            val schema = m.schema // model schema excludes the target variable...
            val dataTupled = Tuple.of(data, schema)
            val tempScore = new Array[Double](2) // 2 classes
            m.predict(dataTupled, tempScore)
            tempScore(1)
        }
    }
    def predictProba(data: Double): Double = model match {
        case m: IsotonicRegressionScaling => m.predict(data)
    }
}


object SmileKitLearn {
    implicit def logitToLogit(m: LogisticRegression): SmileKitLearn[LogisticRegression] =
        new SmileKitLearn[LogisticRegression](m)
    implicit def rfToRf(m: RandomForest): SmileKitLearn[RandomForest] =
        new SmileKitLearn[RandomForest](m)
}



class SmileFrame(data: Array[Transaction]){
    /* Converts an Array[Array[AnyVal]] to a Smile DataFrame. 
    
    Achieved by saving the array to CSV and loading it back. That's not great, but it's all I have
    because doing in-memory conversion throws unreadable errors. 
    
    I can convert the array to a Smile Tuple and apply ofDataFrame() but it throws an error
    when the Tuples are of mixed primitive types, which is a requirement here.
    */
    def asDataFrame(schema: StructType, rootPath: String): DataFrame = {
        val format = CSVFormat.DEFAULT.withFirstRecordAsHeader()

        def caseclassToString(tr: Transaction) = tr match {
            case Transaction(la, am, cnt) => f"$la, $am%1.2f, $cnt%1.2f"
            case _ => "not a transaction."
        }

        //if (schema.fields.size != data.size)
        //throw new Exception("Number of fields don't match. Check the schema.")

        val stringified: Array[String] = for {
            row <- data
        } yield caseclassToString(row)
        
        val fields = schema.fields.map(_.name).mkString(",")

        writeFile(s"$rootPath/temp.csv", stringified, fields)

        val df: DataFrame = Read.csv(s"$rootPath/temp.csv", format, schema)
        df
    }
}

object SmileFrame {
    implicit def ArrayToDataFrame(d: Array[Transaction]): SmileFrame =
        new SmileFrame(d)
}