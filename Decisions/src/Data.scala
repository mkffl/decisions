package decisions

import probability_monad._
import scala.util
import smile.classification._

object TransactionsData {
    case class Transaction(UserType: Int, amount: Double, count: Double)

    def discreteFromBeta(values: Iterable[Int], beta: probability_monad.Distribution[Double]): Distribution[Int] = {
        val vec: Vector[Int] = values.toVector
        beta.map(x => (x * vec.length).toInt)
    }

    // https://math.stackexchange.com/questions/2149570/how-to-generate-sample-from-bimodal-distribution/2149644
    def shiftedPositiveGaussian(loc: Double): Distribution[Double] = {
        for {
        x <- Distribution.normal
        } yield math.abs((x + loc))
    }

    // Should put these in a config file
    val p_w1 = 0.3

    val betaFraud = Distribution.beta(0.5, 0.4)

    val betaRegular = Distribution.beta(6, 6)

    val transactionCountFraud = discreteFromBeta(1 to 15, betaFraud)

    val transactionCountRegular = discreteFromBeta(1 to 15, betaRegular)

    val fraudAmount = shiftedPositiveGaussian(4)

    val regularAmount = shiftedPositiveGaussian(3)

    def transactionAmount(userType: Int) = userType match {
        case 0 => regularAmount
        case 1 => fraudAmount
    }

    def transactionCount(userType: Int) = userType match {
        case 0 => transactionCountRegular
        case 1 => transactionCountFraud
    }

    def transaction: probability_monad.Distribution[Transaction] = {
        for {
            userType <- Distribution.bernoulli(p_w1)
            amount <- transactionAmount(userType)
            count <- transactionCount(userType)
        } yield Transaction(userType, amount, count)
    }
}