/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.objective;

import com.parallax.ml.instance.Instance;
import com.parallax.ml.model.Model;
import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * interface for scoring functions based on
 * objective values
 * 
 * for instance, mean, meadian, etc.
 *
 * @param <T> the generic type
 * @author jattenberg
 */
public interface ObjectiveScorer<T extends Target>  {

	/**
	 * add consideration to the performance of the current model
	 * on the supplied instances.
	 *
	 * @param <I> the generic type
	 * @param <E> the element type
	 * @param <M> the generic type
	 * @param instances the instances
	 * @param model the model
	 * @return computed objective value on the input instances only
	 */
	public <I extends Instance<T>, E extends Iterable<I>, M extends Model<T, M>> double evaluate(E instances, M model);

	/**
	 * incorporates a single objective score from some Objective object.
	 *
	 * @param partialScore the partial score
	 * @return input objective
	 */
	public double evaluate(double partialScore);

	
	/**
	 * returns a double representing the performance of the model and instances
	 * observed during calls to evaluate
	 * 
	 * generally, higher is better.
	 *
	 * @return the score
	 */
	public double getScore();
	
	/**
	 * reset any internal data structures;
	 * fresh computation for the next batch.
	 */
	public void reset();
	
}
