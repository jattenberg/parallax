/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.evaluation.ConfusionMatrix;
import com.parallax.ml.target.BinaryClassificationTarget;

/**
 * Binary Target decision tree split criterion that analyzes the best possible
 * accuracy (% of correctly classified examples) possible using a split.
 * 
 * @see {@link <a href="http://en.wikipedia.org/wiki/Accuracy">Accuracy and Precision</a>}
 * 
 * @author jattenberg
 */
public class BinaryTargetAccuracySplitCriterion implements
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
		ConfusionMatrix confPos = new ConfusionMatrix(2);
		ConfusionMatrix confNeg = new ConfusionMatrix(2);

		for (AttributeValueLabel<BinaryClassificationTarget> avl : attributeValues
				.keySet()) {
			double target = avl.getLabel().getValue();
			double threshSide = avl.getValue() > split ? 1 : 0;

			confPos.addInfo(target, threshSide);
			confNeg.addInfo(target, 1 - threshSide);
		}

		return Math.max(confPos.computeAccuracy(), confNeg.computeAccuracy());
	}

}
