package com.dsi.parallax.optimization.stochastic.anneal;

/**
 * Base class for setting and using a single learning rate and schedule along all the 
 * dimensions of the data 
 * 
 * @author jattenberg
 *
 */
public abstract class DimensionAgnosticAnnealingSchedule extends AnnealingSchedule {


	public DimensionAgnosticAnnealingSchedule(double initialLearningRate) {
		super(initialLearningRate);
	}

	@Override
	public double learningRate(int epoch, int dimension) {
		return learningRate(epoch);
	}
	
	/**
	 * Returns the current global learning rate for all the dimensions based 
	 * on the current epoch
	 * @param epoch
	 * 			the current epoch
	 */
	abstract public double learningRate(int epoch);

}
