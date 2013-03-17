/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection.featureselection;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import com.parallax.ml.distributions.UnivariateGaussianDistribution;
import com.parallax.ml.instance.Instance;
import com.parallax.ml.instance.Instances;
import com.parallax.ml.util.pair.PrimitivePair;
import com.parallax.ml.util.pair.SecondDescendingComparator;

/**
 * Feature Selection based on the variance of values; selects those values with
 * the highest variance.
 */
public class GreatestVarianceFeatureSelection extends AbstractFeatureSelection {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8209303961012785845L;

	/**
	 * Instantiates a new greatest variance feature selection.
	 * 
	 * @param inDim
	 *            the dimension of the incoming problem space
	 * @param numToKeep
	 *            the number of feature values to ckeep.
	 */
	public GreatestVarianceFeatureSelection(int inDim, int numToKeep) {
		super(inDim, numToKeep);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.projection.featureselection.FeatureSelection#build(com
	 * .parallax.ml.instance.Instances)
	 */
	@Override
	public void build(Instances<?> instances) {
		UnivariateGaussianDistribution[] gaussians = new UnivariateGaussianDistribution[inDim];
		for (Instance<?> inst : instances) {
			for (int x_i : inst) {
				if (null == gaussians[x_i])
					gaussians[x_i] = new UnivariateGaussianDistribution();
				double y_i = inst.getFeatureValue(x_i);
				gaussians[x_i].observe(y_i);
			}
		}

		List<PrimitivePair> featureVariancePairs = Lists.newArrayList();
		for (int i = 0; i < inDim; i++) {
			if (null == gaussians[i])
				featureVariancePairs.add(new PrimitivePair(i, 0));
			else
				featureVariancePairs.add(new PrimitivePair(i, gaussians[i]
						.getVariance()));
		}

		Collections
				.sort(featureVariancePairs, new SecondDescendingComparator());
		for (int i = 0; i < inDim; i++)
			selectedFeatures[i] = (int) featureVariancePairs.get(i).getFirst();
	}

}
