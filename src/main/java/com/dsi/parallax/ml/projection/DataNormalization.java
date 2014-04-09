package com.dsi.parallax.ml.projection;

import java.util.Collection;
import java.util.Map;

import org.apache.commons.math.stat.descriptive.SummaryStatistics;

import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.pair.PrimitivePair;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.google.common.collect.Maps;


/**
 * Class for Normalizing the data. It computes and stores the means and 
 * standard deviation of all the features and uses them to normalize 
 * each feature. This involves subtracting the mean of each feature 
 * from its value and dividing the result by the standard deviation
 *  
 * @author spchopra
 *
 */

public class DataNormalization extends AbstractConstructedProjection {

	/** The Constant serialVersionUID. */
    private static final long serialVersionUID = 2910854500392344039L;

	/** The map which stores the mean and standard deviation */
    private Map<Integer, PrimitivePair> meanStdDevPairs;
    
    /** The is trained. */
    private boolean isTrained = false;
       
    /**
     * Instantiates a new DataNormalization object
     *
     * @param inDim the input dimension of the data
     */
    public DataNormalization(int inDim) {
    	super(inDim, inDim);
    	meanStdDevPairs = Maps.newHashMap();
    }
    
    /**
     * Instantiates a new DataNormalization object
     *
     * @param inDim the input dimension of the data
     * @param X the List of data samples to be normalized
     */
    public DataNormalization(int inDim, Collection<LinearVector> X) {
    	super(inDim, inDim);
        meanStdDevPairs = Maps.newHashMap();
        build(X);
    }
    
    /**
     * Instantiates a new DataNormalization object
     *
     * @param inDim the input dimension of the data
     * @param X the collection of Instances
     */
    public DataNormalization(int inDim, Instances<?> X) {
        this(inDim, X.getFeatureVectors());
    }
    
    /**
     * Computes the mean and standard deviation of the input data
     */
    @Override
    public void build(Collection<LinearVector> X) {
    	Map<Integer, SummaryStatistics> stats = Maps.newHashMap();
        // loop over the linear vectors and collect the feature values
        for(LinearVector vect : X) {
        	for (int i = 0; i < vect.size(); i++) {
        		if (!stats.containsKey(i)) { 
        			stats.put(i,  new SummaryStatistics());
        		}
        		stats.get(i).addValue(vect.getValue(i));
        	}
        }
        // compute the mean and standard deviation of the data        
        for (int i : stats.keySet()) {
        	double mean = stats.get(i).getMean();
        	double stdDev = stats.get(i).getStandardDeviation();
        	meanStdDevPairs.put(i, new PrimitivePair(mean, stdDev));
        }
        isTrained = true;
    }
    
    /** 
     * Returns a boolean indicating whether the mean and standard 
     * deviation of the data have been computed
     */
    @Override
    public boolean isBuilt() {
        return isTrained;
    }

    /**
     * Normalized the given input features by subtracting the 
     * stored mean and dividing the standard deviation from the 
     * feature values
     * 
     * @param input the input features to be be normalized
     */
    @Override
    public LinearVector project(LinearVector input) {
    	LinearVector output = LinearVectorFactory.getVector(input.size());
    	
    	for (int i = 0; i < output.size(); i++) { 
    		double value = input.getValue(i);
    		if (!meanStdDevPairs.containsKey(i)) { 
    			output.resetValue(i, value);
    		} else {
    			PrimitivePair meanStdDevPair = meanStdDevPairs.get(i);
    			double norm_value = (value - meanStdDevPair.first)/meanStdDevPair.second;
    			output.resetValue(i, MLUtils.floatingPointEquals(
    						meanStdDevPair.second, 0) ? value : norm_value);
    		}
    	}
    	return output;
    }
        
    
        
    /**
     * Prints the stored mean and standard deviation
     */
    @Override
    public String toString() {
    	String means = "Means: ";
    	String stds = "Standard Deviation: ";
    	String output = "";
    	
    	for (int i = 0; i < inDim; i++) {
    		PrimitivePair meanStdDevPair = meanStdDevPairs.get(i);
    		means = means + " " + Double.toString(meanStdDevPair.first);
    		stds = stds + " " + Double.toString(meanStdDevPair.second);
    	}
    	output = output + means + "\n" + stds;
    	
    	return output;
    }	
}	
        
    
    