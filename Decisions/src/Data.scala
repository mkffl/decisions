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

    def gaussianFeature(loc: Double)(userType: Int) = userType match {
        case 0 => Distribution.normal + loc
        case 1 => Distribution.normal - loc
    }

    def feat1 = gaussianFeature(1.0)(_)
    def feat2 = gaussianFeature(1.5)(_)
    def feat3 = gaussianFeature(0.8)(_)
    def feat4 = gaussianFeature(-2.0)(_)
    def feat5 = gaussianFeature(-1.0)(_)

    case class Transact(userType: Int, 
        f1: Double, f2: Double, f3: Double, f4: Double, f5: Double, 
        f6: Double, f7: Double, f8: Double, f9: Double, f10: Double
    )

    def transact(p_w1: Double): probability_monad.Distribution[Transact] = {
        for {
            ut <- Distribution.bernoulli(p_w1)
            f1 <- feat1(ut)
            f2 <- feat2(ut)
            f3 <- feat3(ut)
            f4 <- feat4(ut)
            f5 <- feat5(ut)
            f6 = f1*1.5 - 3.0
            f7 = f2*2.0 - 4.0
            f8 = f3*0.7 + 1.8
            f9 = f4*0.5 - 0.5
            f10 = f5 + 3.0
        } yield Transact(ut,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10)
    }
}