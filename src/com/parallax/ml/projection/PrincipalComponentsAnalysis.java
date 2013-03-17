/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection;

import java.util.Collection;

import org.apache.commons.lang.time.StopWatch;
import org.apache.log4j.Logger;
import org.ejml.alg.dense.decomposition.DecompositionFactory;
import org.ejml.alg.dense.decomposition.SingularValueDecomposition;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;
import org.ejml.ops.SingularOps;

import com.parallax.ml.instance.Instances;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;

// TODO: Auto-generated Javadoc
/**
 * projection.
 *
 * @author jattenberg
 */
public class PrincipalComponentsAnalysis extends AbstractConstructedProjection {

    /** The Constant logger. */
    private static final Logger logger = Logger.getLogger(PrincipalComponentsAnalysis.class);
    
    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -3429955243265359259L;
    
    /** The V_t. */
    private DenseMatrix64F V_t;
    
    /** The is trained. */
    private boolean isTrained = false;
    
    /** The mean. */
    private double mean[];
    
    /** The a. */
    private transient DenseMatrix64F A = new DenseMatrix64F(1,1);
    
    /** The sample index. */
    private int sampleIndex;
    
    /**
     * Instantiates a new principal components analysis.
     *
     * @param inDim the in dim
     * @param numComponents the num components
     */
    public PrincipalComponentsAnalysis(int inDim, int numComponents) {
    	super(inDim, numComponents);
    }
    
    /**
     * Instantiates a new principal components analysis.
     *
     * @param inDim the in dim
     * @param numComponents the num components
     * @param X the x
     */
    public PrincipalComponentsAnalysis(int inDim, int numComponents, Collection<LinearVector> X) {
        super(inDim, numComponents);
        build(X);
    }
    
    /**
     * Instantiates a new principal components analysis.
     *
     * @param inDim the in dim
     * @param numComponents the num components
     * @param X the x
     */
    public PrincipalComponentsAnalysis(int inDim, int numComponents, Instances<?> X) {
        this(inDim, numComponents, X.getFeatureVectors());
    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.ConstructedProjection#build(java.util.Collection)
     */
    @Override
    public void build(Collection<LinearVector> X) {
        logger.info("adding " + X.size() + " examples to internal data structure for SVD");
        StopWatch sw = new StopWatch();
        sw.start();
        setup(X.size(), inDim);
        for(LinearVector x : X) {
            addSample(x.getW());
        }
        logger.info("done, took: " + sw.getTime() + "ms. performing SVD transform");
        sw.reset();
        sw.start();
        computeBasis(outDim);
        A = null;
        isTrained = true;
        logger.info("done, took: " + sw.getTime() + "ms");

    }
    
    /* (non-Javadoc)
     * @see com.parallax.ml.projection.ConstructedProjection#isBuilt()
     */
    @Override
    public boolean isBuilt() {
        return isTrained;
    }

    /* (non-Javadoc)
     * @see com.parallax.ml.projection.Projection#project(com.parallax.ml.util.vector.LinearVector)
     */
    @Override
    public LinearVector project(LinearVector x) {
        double[] proj = sampleToEigenSpace(x.getW());
        LinearVector out = LinearVectorFactory.getVector(proj);
        return out;
    }
    
    /**
     * Must be called before any other functions. Declares and sets up internal data structures.
     *
     * @param numSamples Number of samples that will be processed.
     * @param sampleSize Number of elements in each sample.
     */
    public void setup( int numSamples , int sampleSize ) {
        mean = new double[ sampleSize ];
        A.reshape(numSamples,sampleSize,false);
        sampleIndex = 0;
        
    }
    
    /**
     * Adds a new sample of the raw data to internal data structure for later processing.  All the samples
     * must be added before computeBasis is called.
     *
     * @param sampleData Sample from original raw data.
     */
    public void addSample( double[] sampleData ) {
        if( inDim != sampleData.length )
            throw new IllegalArgumentException("Unexpected sample size");
        if( sampleIndex >= A.getNumRows() )
            throw new IllegalArgumentException("Too many samples");

        for( int i = 0; i < sampleData.length; i++ ) {
            A.set(sampleIndex,i,sampleData[i]);
        }
        sampleIndex++;
    }
    
    /**
     * Computes a basis (the principle components) from the most dominant eigenvectors.
     *
     * @param numComponents Number of vectors it will use to describe the data.  Typically much
     * smaller than the number of elements in the input vector.
     */
    public void computeBasis( int numComponents ) {
        if( numComponents > inDim )
            throw new IllegalArgumentException("More components requested that the data's length.");
        if( sampleIndex != A.getNumRows() )
            throw new IllegalArgumentException("Not all the data has been added");
        if( numComponents > sampleIndex )
            throw new IllegalArgumentException("More data needed to compute the desired number of components");

        // compute the mean of all the samples
        for( int i = 0; i < A.getNumRows(); i++ ) {
            for( int j = 0; j < mean.length; j++ ) {
                mean[j] += A.get(i,j);
            }
        }
        for( int j = 0; j < mean.length; j++ ) {
            mean[j] /= A.getNumRows();
        }

        // subtract the mean from the original data
        for( int i = 0; i < A.getNumRows(); i++ ) {
            for( int j = 0; j < mean.length; j++ ) {
                A.set(i,j,A.get(i,j)-mean[j]);
            }
        }

        // Compute SVD and save time by not computing U
        SingularValueDecomposition<DenseMatrix64F> svd = 
                DecompositionFactory.svd(A.numRows, A.numCols, false, true, false);
        if( !svd.decompose(A) )
            throw new RuntimeException("SVD failed");

        V_t = svd.getV(true);
        DenseMatrix64F W = svd.getW(null);

        // Singular values are in an arbitrary order initially
        SingularOps.descendingOrder(null,false,W,V_t,true);

        // strip off unneeded components and find the basis
        V_t.reshape(numComponents,mean.length,true);
    }
    
    /**
     * Converts a vector from sample space into eigen space.
     *
     * @param sampleData Sample space data.
     * @return Eigen space projection.
     */
    public double[] sampleToEigenSpace( double[] sampleData ) {
        if( sampleData.length != inDim )
            throw new IllegalArgumentException("Unexpected sample length");
        DenseMatrix64F mean = DenseMatrix64F.wrap(inDim,1,this.mean);

        DenseMatrix64F s = new DenseMatrix64F(inDim,1,true,sampleData);
        DenseMatrix64F r = new DenseMatrix64F(outDim,1);

        CommonOps.sub(s,mean,s);

        CommonOps.mult(V_t,s,r);

        return r.data;
    }

    /* (non-Javadoc)
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        return V_t.toString();
    }
    
    /**
     * Converts a vector from eigen space into sample space.
     *
     * @param eigenData Eigen space data.
     * @return Sample space projection.
     */
    public double[] eigenToSampleSpace( double[] eigenData ) {
        if( eigenData.length != outDim )
            throw new IllegalArgumentException("Unexpected sample length");

        DenseMatrix64F s = new DenseMatrix64F(inDim,1);
        DenseMatrix64F r = DenseMatrix64F.wrap(outDim,1,eigenData);
        
        CommonOps.multTransA(V_t,r,s);

        DenseMatrix64F mean = DenseMatrix64F.wrap(inDim,1,this.mean);
        CommonOps.add(s,mean,s);

        return s.data;
    }


    /**
     * <p>
     * The membership error for a sample.  If the error is less than a threshold then
     * it can be considered a member.  The threshold's value depends on the data set.
     * </p>
     * <p>
     * The error is computed by projecting the sample into eigenspace then projecting
     * it back into sample space and
     * </p>
     * 
     * @param sampleA The sample whose membership status is being considered.
     * @return Its membership error.
     */
    public double errorMembership( double[] sampleA ) {
        double[] eig = sampleToEigenSpace(sampleA);
        double[] reproj = eigenToSampleSpace(eig);


        double total = 0;
        for( int i = 0; i < reproj.length; i++ ) {
            double d = sampleA[i] - reproj[i];
            total += d*d;
        }

        return Math.sqrt(total);
    }

    /**
     * Computes the dot product of each basis vector against the sample.  Can be used as a measure
     * for membership in the training sample set.  High values correspond to a better fit.
     *
     * @param sample Sample of original data.
     * @return Higher value indicates it is more likely to be a member of input dataset.
     */
    public double response( double[] sample ) {
        if( sample.length != inDim )
            throw new IllegalArgumentException("Expected input vector to be in sample space");

        DenseMatrix64F dots = new DenseMatrix64F(outDim,1);
        DenseMatrix64F s = DenseMatrix64F.wrap(inDim,1,sample);

        CommonOps.mult(V_t,s,dots);

        return NormOps.normF(dots);
    }
}
