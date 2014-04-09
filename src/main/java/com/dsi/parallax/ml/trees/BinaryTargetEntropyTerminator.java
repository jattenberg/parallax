/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;

/**
 * a terminator for binary decision trees that is useful for ensuring that there
 * must be some minimum amount of entropy in the label distribution of the nodes
 * present in a branch.
 * 
 * @author jattenberg
 */
public class BinaryTargetEntropyTerminator implements
		Terminator<BinaryClassificationTarget> {

	/** The minimum acceptable entropy. */
	private final double minEntropy;

	/**
	 * Instantiates a new binary target entropy terminator.
	 * 
	 * @param minEntropy
	 *            the minimum acceptable entropy
	 */
	public BinaryTargetEntropyTerminator(double minEntropy) {
		this.minEntropy = minEntropy;
		checkArgument(minEntropy >= 0,
				"entropy must be non-negative. given: %s", minEntropy);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.trees.Terminator#terminate(com.parallax.ml.instance.Instances
	 * , int)
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> boolean terminate(
			I instances, int depth) {
		double[] counts = new double[2];

		for (Instance<BinaryClassificationTarget> inst : instances) {
			if (inst.getLabel().getValue() > 0.5)
				counts[0]++;
			else
				counts[1]++;
		}
		return MLUtils.entropy(counts) < minEntropy;
	}

}
