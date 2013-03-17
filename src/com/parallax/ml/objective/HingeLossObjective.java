/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.objective;

import com.parallax.ml.evaluation.HingeLoss;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.model.Model;
import com.parallax.ml.target.BinaryClassificationTarget;

/**
 * L1 hinge loss: max(0, 1-y*f(x)) where y and f(x) are on the svm scale [-1, 1]
 * 
 * this objective averages this score over all instances.
 * 
 * @author jattenberg
 */
public class HingeLossObjective implements
		Objective<BinaryClassificationTarget> {

	/** Hinge loss used for computing objective values */
	private static final HingeLoss hinge = new HingeLoss(1.);

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.objective.Objective#evaluate(java.lang.Iterable,
	 * com.parallax.ml.model.Model)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>, E extends Iterable<I>, M extends Model<BinaryClassificationTarget, M>> double evaluate(
			E instances, M model) {
		double total = 0;
		double count = 0;
		for (I inst : instances) {
			count++;
			total += hinge.loss(inst.getLabel(), model.predict(inst));
		}
		return count == 0 ? 0 : total / count;
	}

}
