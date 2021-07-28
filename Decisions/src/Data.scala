package decisions

import probability_monad._
import scala.util
import smile.classification._

object TransactionsData extends decisions.Shared.LinAlg{
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


    def shiftCentroid(ut: User, cluster: Cluster, base: Row): Row = (ut, cluster) match {
        case (Regular, Cluster1) => addVectors(f1,base)
        case (Regular, Cluster2) => addVectors(f2,base)
        case (Fraudster, Cluster1) => addVectors(f3,base)
        case (Fraudster, Cluster2) => addVectors(f4,base)
    }

    def repeat(base: Row): Row = dot(Matrix(base),redundant).head

    def transact(p_w1: Double): probability_monad.Distribution[Transaction] = {
        for {
            userDraw <- Distribution.bernoulli(p_w1)
            ut = if (userDraw==1){Fraudster} else Regular
            cluster <- Distribution.bernoulli(0.5)
            cl = if (cluster==1){Cluster1} else Cluster2
            f1 <- Distribution.normal
            f2 <- Distribution.normal
            f3 <- Distribution.normal
            f4 <- Distribution.normal
            f5 <- Distribution.normal
            gaussianValues = Vector(f1,f2,f3,f4,f5)
            base = shiftCentroid(ut, cl, gaussianValues) // Vector of size 5
            repeats = repeat(base)
            features = base ++ repeats // Vector of size 10
        } yield Transaction(ut,features)
    }



}