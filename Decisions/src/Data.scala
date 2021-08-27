package decisions

import probability_monad._
import scala.util
import smile.classification._

object TransactionsData extends decisions.Shared.LinAlg{
    import RowLinAlg._, MatLinAlg._   
    /*
        f1    f2   f3   f4   f5
    array([[ 1., -1., -1., -1.,  1.], --w0
           [-1.,  1., -1.,  1., -1.], --w0
           [ 1.,  1.,  1.,  1.,  1.], --w1
           [-1., -1.,  1., -1., -1.]]) --w1
    */
    sealed trait User
    object Regular extends User
    object Fraudster extends User
    
    trait Cluster
    object Cluster1 extends Cluster
    object Cluster2 extends Cluster
    
    case class Transaction(UserType: User, features: Row)
    
    val f1: Row = Vector(1,-1,-1,-1,1)
    val f2: Row = Vector(-1,1,-1,1,-1)
    val f3: Row = Vector(1,1,1,1,1)
    val f4: Row = Vector(-1,-1,1,-1,-1)
    
    val r1 = Row(-0.88332417,  0.126571  , -0.15694154,  0.07552767,  0.29527662)
    val r2 = Row(-0.08826325, -0.74441347, -0.05989755, -0.97603216,  0.02567397)
    val r3 = Row(-0.65018744,  0.09665695,  0.89011513,  0.14527666,  0.42994935)
    val r4 = Row(-0.16310978, -0.04414065, -0.60458077, -0.2222262 , -0.67554751)
    val r5 = Row( 0.93812198, -0.78866268,  0.0177584 ,  0.99954504, -0.8185286)
    val redundant = Matrix(r1,r2,r3,r4,r5)


    def shiftCentroid(ut: User, cluster: Cluster, noise: Row): Row = (ut, cluster) match {
        case (Regular, Cluster1) => f1 + noise
        case (Regular, Cluster2) => f2 + noise
        case (Fraudster, Cluster1) => f3 + noise
        case (Fraudster, Cluster2) => f4 + noise
    }

    /* Linear transformation of informative features */
    def repeat(base: Row): Row = (Matrix(base) at redundant).head

    def transact(p_w1: Double): probability_monad.Distribution[Transaction] = for {
            userDraw <- Distribution.bernoulli(p_w1)
            ut = if (userDraw==1){Fraudster} else Regular
            cluster <- Distribution.bernoulli(0.5)
            cl = if (cluster==1){Cluster1} else Cluster2
            g1 <- Distribution.normal
            g2 <- Distribution.normal
            g3 <- Distribution.normal
            g4 <- Distribution.normal
            g5 <- Distribution.normal
            gaussian = Vector(g1,g2,g3,g4,g5)
            informative = shiftCentroid(ut, cl, gaussian) // Vector of size 5
            repeats = repeat(informative)
            features = informative ++ repeats // Vector of size 10
        } yield Transaction(ut,features)



}