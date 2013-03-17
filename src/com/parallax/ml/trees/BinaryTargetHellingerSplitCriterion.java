/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.target.BinaryClassificationTarget;

/**
 * A binary label tree decision function learner that learns a vector value
 * threshold function using the Hellinger distance. The Hellinger distance is
 * supposed to be less sensitive to skewed data than the more well known
 * information gain method. It also behaves about the same as information gain
 * on balanced data. Thus, it is thought that the Hellinger method may be
 * superior to information gain.
 * 
 * For a given split (sets X and Y) for two categories (a and b) d(X, Y) = sqrt(
 * (sqrt(Xa / Na) - sqrt(Xb / Nb))^2 + (sqrt(Ya / Na) - sqrt(Yb / Nb))^2) 
 * 
 * where
 * Xa = number of a's in X, Xb = number of b's in X, Ya = number of a's in Y, Yb
 * = number of b's in Y, Na = total number of a's (= Xa + Ya), and Nb = total
 * number of b's (= Xb + Yb).
 * 
 * The Hellinger distance ranges between 0 and sqrt(2), inclusive.
 */
public class BinaryTargetHellingerSplitCriterion implements
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

		double[][] posteriors = new double[2][2];
		double[] xCounts = new double[2];

		for (AttributeValueLabel<BinaryClassificationTarget> avl : attributeValues
				.keySet()) {
			int index = avl.getLabel().getValue() > 0.5 ? 1 : 0;
			int threshSide = avl.getValue() > split ? 1 : 0;

			posteriors[threshSide][index] += attributeValues.get(avl);
			xCounts[threshSide] += attributeValues.get(avl);
		}

		double hellingerSum = 0.0;

		hellingerSum += Math.pow(Math.sqrt(posteriors[1][0] / xCounts[0])
				- Math.sqrt(posteriors[1][1] / xCounts[1]), 2);
		hellingerSum += Math.pow(Math.sqrt(posteriors[0][0] / xCounts[0])
				- Math.sqrt(posteriors[0][1] / xCounts[1]), 2);

		return Math.sqrt(hellingerSum);

	}

}
