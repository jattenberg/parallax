/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.model;

import java.io.Serializable;

import com.parallax.ml.target.Target;

/**
 * The Class AbstractModel. Base class for all predictive models
 *
 * @param <T> the generic type
 * @param <M> the generic type
 */
public abstract class AbstractModel<T extends Target, M extends AbstractModel<T, M>>
		implements Model<T, M>, Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 2914734744627347537L;
	
	/** The dimension; number of features in the model */
	protected final int dimension;
	
	/** does the model have a bias term? if set to false, the model's hyperplane will pass through the origin */
	protected final boolean bias;
	
	/** The model itself. used for method chaining only. */
	protected final M model;

	/**
	 * Instantiates a new abstract model.
	 *
	 * @param dimension the number of features in the instantiated classifier
	 * @param bias should the model have an additional (+1) intercept term? 
	 */
	protected AbstractModel(int dimension, boolean bias) {
		this.bias = bias;
		this.dimension = bias ? dimension + 1 : dimension;
		this.model = getModel();
	}

	/**
	 * How many features does the model have?
	 * @return the number of dimensions in the model.
	 */
	@Override
	public int getModelDimension() {
		return dimension;
	}

	/**
	 * does the model have an additional (+1) bias term? 
	 * @return returns true if the model has an additional bias term.
	 */
	@Override
	public boolean usesBiasTerm() {
		return bias;
	}

	/**
	 * Gets the model. All concrete implementations 
	 * should implement and (return this)
	 *
	 * @return the model itself. Used for method chaining.
	 */
	protected abstract M getModel();
}
