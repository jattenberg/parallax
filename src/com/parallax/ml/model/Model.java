/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.model;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.target.Target;

/**
 * base interface for predictive models.
 *
 * @param <T> The type of target being predicted
 * @param <M> The concrete type of the model, used for method chaining
 * @author jattenberg
 */
public interface Model<T extends Target, M extends Model<T,M>> {
	
	/**
	 * Predict - guess the target variable for the input instance
	 *
	 * @param instance the instance being labeled
	 * @return predicted target value
	 */
	public abstract T predict(Instance<?> instance);
	
	/**
	 * Train the model on a set of labeled instances
	 *
	 * @param <I> the generic type of labeled instances being considered.
	 * @param instances labeled training data
	 */
	public <I extends Instances<? extends Instance<T>>> void train(I instances);
	
	/**
	 * Gets the model dimension.
	 *
	 * @return the model dimension
	 */
	public int getModelDimension();	
	
	/**
	 * Initialize the data structures of the model,
	 * make ready for training.
	 *
	 * @return the prediction model
	 */
	public M initialize();
	
	/**
	 * Uses bias term.
	 *
	 * @return true, if successful
	 */
	public boolean usesBiasTerm();
}
