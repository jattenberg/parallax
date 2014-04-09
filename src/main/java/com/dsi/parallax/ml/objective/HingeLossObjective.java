/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.objective;

import com.dsi.parallax.ml.evaluation.HingeLoss;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.model.Model;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;

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
