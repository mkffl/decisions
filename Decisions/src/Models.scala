package decisions
import smile.classification._
import smile.data.{Tuple, DataFrame}
import smile.data.`type`._
import smile.io.Read

import org.apache.commons.csv.CSVFormat

import decisions.Shared._, LinAlg._, Stats._, FileIO._, RowLinAlg._,
MatLinAlg._, CollectionsStats._
import decisions.TransactionsData._
import java.util.Properties

object Dataset {
  trait DecisionDataset
  trait RecognizerDataset extends DecisionDataset
  trait CalibrationDataset extends DecisionDataset
  trait EvaluationDataset extends DecisionDataset
}

trait Systems {
  trait Transformer
  trait Recognizer extends Transformer
  trait Calibrator extends Transformer

  case class Logit(name: String, params: Option[Properties]) extends Recognizer
  case class RF(name: String, params: Option[Properties]) extends Recognizer
  case class SupportVectorMachine(name: String, params: Option[Properties])
      extends Recognizer
  case class Isotonic(name: String) extends Calibrator
  case class Platt(name: String) extends Calibrator
  case class Bionomial(name: String) extends Calibrator // what's that?
  case object Uncalibrated extends Calibrator

  type System = Tuple2[Recognizer, Calibrator]
}

class SmileKitLearn[T](model: T) {
  def predictproba(data: Array[Array[Double]]): Array[Double] = model match {
    case m: LogisticRegression => {
      val tempScore = new Array[Double](2) // 2 classes
      data.map { x =>
        m.predict(x, tempScore)
        tempScore(1)
      }
    }
    case m: RandomForest => {
      val schema = m.schema // model schema excludes the target variable...
      val dataTupled = for (row <- data) yield Tuple.of(row, schema)
      val tempScore = new Array[Double](2) // 2 classes
      dataTupled.map { x =>
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
    case m: LogisticRegression => {
      val tempScore = new Array[Double](2) // 2 classes
      m.predict(data, tempScore)
      tempScore(1)
    }
    case m: RandomForest => {
      val schema = m.schema // model schema excludes the target variable...
      val dataTupled = Tuple.of(data, schema)
      val tempScore = new Array[Double](2) // 2 classes
      m.predict(dataTupled, tempScore)
      tempScore(1)
    }
    case m: SVM[Array[Double]] => expit(m.score(data))
  }
  def predictProba(data: Double): Double = model match {
    case m: IsotonicRegressionScaling => m.predict(data)
  }
}

object SmileKitLearn {
  implicit def logitToLogit(
      m: LogisticRegression
  ): SmileKitLearn[LogisticRegression] =
    new SmileKitLearn[LogisticRegression](m)
  implicit def rfToRf(m: RandomForest): SmileKitLearn[RandomForest] =
    new SmileKitLearn[RandomForest](m)
  implicit def svmToSVM(
      m: SVM[Array[Double]]
  ): SmileKitLearn[SVM[Array[Double]]] =
    new SmileKitLearn[SVM[Array[Double]]](m)
}

class SmileFrame(data: Array[Transaction]) {
  /* Converts an Array[Array[AnyVal]] to a Smile DataFrame.

    Achieved by saving the array to CSV and loading it back. That's not great, but it's all I have
    because doing in-memory conversion throws unreadable errors.

    I can convert the array to a Smile Tuple and apply ofDataFrame() but it throws an error
    when the Tuples are of mixed primitive types, which is a requirement here.
   */

  /* ","-separated string representation of a Vector[Double]
        Examples: Vector(3.30, 2.40) => "3.30,2.40"
   */

  def asDataFrame(schema: StructType, rootPath: String): DataFrame = {
    val format = CSVFormat.DEFAULT.withFirstRecordAsHeader()

    // add as argument of type Transaction => String
    def caseclassToString(tr: Transaction) = tr match {
      case Transaction(user, features) =>
        s"${userToInt(user)}," + stringifyVector(features)
      case _ => "not a transaction."
    }

    def userToInt(user: User): Int = user match {
      case Regular   => 0
      case Fraudster => 1
    }

    //if (schema.fields.size != data.size)
    //throw new Exception("Number of fields don't match. Check the schema.")

    val asString: Array[String] = for {
      row <- data
    } yield caseclassToString(row)

    val stringify: (Transaction => String) = row => {
      val startIndex = row.toString.indexOf("(") + 1
      val endIndex = row.toString.indexOf(")")
      row.toString.slice(startIndex, endIndex)
    }

    //val stringified: Array[String] = data.map(toString)

    val fields = schema.fields.map(_.name).mkString(",")

    writeFile(s"$rootPath/temp.csv", asString, fields)

    val df: DataFrame = Read.csv(s"$rootPath/temp.csv", format, schema)
    df
  }
}

object SmileFrame {
  implicit def ArrayToDataFrame(d: Array[Transaction]): SmileFrame =
    new SmileFrame(d)
}
