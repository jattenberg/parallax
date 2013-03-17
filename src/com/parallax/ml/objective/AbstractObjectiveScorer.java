/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.objective;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.model.Model;
import com.parallax.ml.target.Target;

/**
 * Base class for ObjectiveScorers; classes that combine multiple evaluations of the objective function
 * into a single unified metric using cross validation. 
 *
 * @param <T> the type of target the model predicts; the type of label each instance has
 */
public abstract class AbstractObjectiveScorer<T extends Target> implements ObjectiveScorer<T> {

	/** The objective. */
	private final Objective<T> objective;
	
	/**
	 * Instantiates a new abstract objective scorer.
	 *
	 * @param objective the type of objective function to be used in this scorer. For instance AUC. 
	 */
	protected AbstractObjectiveScorer(Objective<T> objective) {
		this.objective = objective;
	}
	
	/* (non-Javadoc)
	 * @see com.parallax.ml.objective.ObjectiveScorer#evaluate(java.lang.Iterable, com.parallax.ml.model.Model)
	 */
	@Override
	public <I extends Instance<T>, E extends Iterable<I>, M extends Model<T, M>> double evaluate(
			E instances, M model) {
		double objVal = objective.evaluate(instances, model);
		evaluate(objVal);
		return objVal;
	}
}
