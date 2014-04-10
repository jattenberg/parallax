/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;

/**
 * Binary Target decision tree split criterion that analyzes the infomration
 * gain provided by a particular split.
 * {@link <a href="http://en.wikipedia.org/wiki/Information_gain_in_decision_trees">Information Gain</a>}
 */
public class BinaryTargetInfoGainSplitCriterion implements
		SplitCriterion<BinaryClassificationTarget> {

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.trees.SplitCriterion#computeObjective(com.parallax.ml
	 * .trees.AttributeValueSet, double)
	 */
	@Override
	public double computeObjective(
			AttributeValueSet<BinaryClassificationTarget> attributeValues,
			double split) {
		double[] priors = new double[2];
		double[][] posteriors = new double[2][2];
		double[] xCounts = new double[2];

		for (AttributeValueLabel<BinaryClassificationTarget> avl : attributeValues
				.keySet()) {
			int index = avl.getLabel().getValue() > 0.5 ? 1 : 0;
			int threshSide = avl.getValue() > split ? 1 : 0;

			priors[index] += attributeValues.get(avl);
			posteriors[threshSide][index] += attributeValues.get(avl);
			xCounts[threshSide] += attributeValues.get(avl);
		}
		double entropy = MLUtils.entropy(priors);
		double posProb = xCounts[1] / (xCounts[1] + xCounts[0]);

		entropy -= posProb * MLUtils.entropy(posteriors[1]);
		entropy -= (1. - posProb) * MLUtils.entropy(posteriors[0]);
		return entropy;
	}
}
