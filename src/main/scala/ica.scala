/**
  * Independent component analysis
  *
  */

package ica


import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.linalg.{
  DenseMatrix => MLDM,
  Matrices => MLMatrices,
  Matrix => MLMatrix,
  Vectors => MLVectors
}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

object ica {

  def pcaWhitening(
      M: IndexedRowMatrix,
      numOfSource: Int
  ): (MLMatrix, IndexedRowMatrix) = {
    val x_svd            = M.computeSVD(k = numOfSource, computeU = false)
    val numSamples       = M.numRows.toInt
    val S: Array[Double] = x_svd.s.toArray
    val V: MLMatrix      = x_svd.V
    val invSqrtLambda    = MLDM.diag(MLVectors.dense(S.map(breeze.numerics.sqrt(numSamples) * 1 / _)))
    val K                = V.multiply(invSqrtLambda)
    val Z                = M.multiply(K)
    (K, Z)
  }

  def fitICA(
      Z: BlockMatrix,
      thresh: Double,
      maxIteration: Int,
      alpha: Double
  ): MLMatrix = {
    // Initialization
    val numSamples: Int  = Z.numCols.toInt
    val numOfSource: Int = Z.numRows.toInt
    val normal01         = breeze.stats.distributions.Gaussian(0, 1)
    val w_init           = BDM.rand(numOfSource, numOfSource, normal01)
    var w                = BDMtoMLDM(sym_decorrelation(w_init))
    var lim              = 999.0
    var i                = 0
    import breeze.linalg._

    // Main ICA loop
    while (i < maxIteration & lim > thresh) {
      val wtx       = Z.transpose.toIndexedRowMatrix.multiply(w)
      val xEntryRdd = wtx.toCoordinateMatrix.entries
      // currently only log cosh non-quadratic function implemented
      val gxRdd = xEntryRdd.map(
        x => MatrixEntry(x.i, x.j, alpha * breeze.numerics.tanh(x.value))
      )
      val gPrimexRdd =
        gxRdd.map(x => MatrixEntry(x.i, x.j, alpha * (1 - x.value * x.value)))
      val g1 = new CoordinateMatrix(gxRdd).toBlockMatrix
      val gPrimeMean = new CoordinateMatrix(gPrimexRdd).toBlockMatrix.blocks
        .map(x => x._2.colIter.map(x => x.toArray.sum / x.size).toArray)
        .collect
        .transpose
        .map(x => x.sum / x.length)
      val G      = (MLDMtoBDM(Z.multiply(g1).toLocalMatrix)) /:/ numSamples.toDouble
      val GPrime = MLDMtoBDM(w)(::, *) *:* BDV(gPrimeMean)
      val w_p    = sym_decorrelation(G - GPrime)
      lim = max(
        breeze.numerics.abs(
          breeze.numerics.abs(breeze.linalg.diag(w_p * MLDMtoBDM(w).t)) - 1.0
        )
      )
      w = BDMtoMLDM(w_p)
      i += 1
    }
    w
  }

  def sym_decorrelation(w: BDM[Double]): BDM[Double] = {
    /*Symmetric decorrelation
     * W <- (W * W.T) ^{-1/2} * W
     */
    val breeze.linalg.svd.SVD(u, s, v) = breeze.linalg.svd(w * w.t)
    u * breeze.linalg.diag(s ^:^ (-0.5)) * v * w
  }

  def MLDMtoBDM(X: MLMatrix): BDM[Double] = {
    val m = BDM.zeros[Double](X.numRows, X.numCols)
    for (i <- 0 until X.numRows) {
      for (j <- 0 until X.numCols) {
        m(i, j) = X.apply(i, j)
      }
    }
    m
  }

  def BDMtoMLDM(X: BDM[Double]): MLMatrix = {
    MLMatrices.dense(X.rows, X.cols, X.toArray)
  }

  // Fit the data with ICA weight matrix
  class ICADfTransform(val w: org.apache.spark.ml.linalg.DenseMatrix) extends Serializable {

    val whitenICATransform: UserDefinedFunction = udf(
      (arr: org.apache.spark.ml.linalg.Vector) => _whitenICATransform(arr, w)
    )

    def _whitenICATransform(
        featureVectors: org.apache.spark.ml.linalg.Vector,
        weightMatrix: org.apache.spark.ml.linalg.DenseMatrix
    ): org.apache.spark.ml.linalg.Vector = {
      weightMatrix.transpose.multiply(featureVectors)
    }
  }
}
