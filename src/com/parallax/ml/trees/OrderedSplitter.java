/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import com.parallax.ml.target.Target;

// TODO: Auto-generated Javadoc
/**
 * order examples according to the value of tine.
 *
 * @param <T> the generic type
 * @author jattenberg
 */
public class OrderedSplitter<T extends Target> implements Splitter<T> {

	/* (non-Javadoc)
	 * @see com.parallax.ml.trees.Splitter#buildSplit(com.parallax.ml.trees.SplitCriterion, com.parallax.ml.trees.AttributeValueSet, int)
	 */
	@Override
	public Benefit buildSplit(SplitCriterion<T> criterion,
			AttributeValueSet<T> attributeValues, int index) {
		T oldLabel = null;
		double bestSplit = Double.NaN;
		double lastValue = Double.NaN;
		double bestSplitBenefit = Double.NEGATIVE_INFINITY;

		// go through all the split values, check their benefit
		for (AttributeValueLabel<T> label : attributeValues.keySet()) {
			double currentValue = label.getValue();
			T currentLabel = label.getLabel();

			if (!Double.isNaN(lastValue) && lastValue != currentValue
					&& !currentLabel.equals(oldLabel)) {
				double currentSplit = (lastValue + currentValue) / 2.0;
				double benefit = criterion.computeObjective(attributeValues,
						currentSplit);

				if (benefit > bestSplitBenefit) {
					bestSplitBenefit = benefit;
					bestSplit = currentSplit;
				}
			}

			lastValue = currentValue;
			oldLabel = currentLabel;
		}
		return new Benefit(index, bestSplitBenefit, bestSplit);
	}
}
