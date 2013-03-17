/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.trees;

import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.trees.BinaryTargetAccuracySplitCriterion;
import com.parallax.ml.trees.BinaryTargetGiniSplitCriterion;
import com.parallax.ml.trees.BinaryTargetHellingerSplitCriterion;
import com.parallax.ml.trees.BinaryTargetInfoGainSplitCriterion;
import com.parallax.ml.trees.SplitCriterion;

/**
 * A simple enumeration over the different binary target decision tree split
 * criteria
 */
public enum BinarySplitCriterionType {

	/** Hellenger Distance. @see {@link BinaryTargetHellingerSplitCriterion } */
	HELLINGER {

		@Override
		public SplitCriterion<BinaryClassificationTarget> buildCriterion() {
			return new BinaryTargetHellingerSplitCriterion();
		}
	},

	/** Gini Index. @see {@link BinaryTargetGiniSplitCriterion} */
	GINI {

		@Override
		public SplitCriterion<BinaryClassificationTarget> buildCriterion() {
			return new BinaryTargetGiniSplitCriterion();
		}
	},

	/** Info gain @see {@link BinaryTargetInfoGainSplitCriterion} */
	INFOGAIN {

		@Override
		public SplitCriterion<BinaryClassificationTarget> buildCriterion() {
			return new BinaryTargetInfoGainSplitCriterion();
		}
	},

	/** Accuracy. @see {@link BinaryTargetAccuracySplitCriterion}. */
	ACCY {

		@Override
		public SplitCriterion<BinaryClassificationTarget> buildCriterion() {
			return new BinaryTargetAccuracySplitCriterion();
		}

	};

	/**
	 * Builds the criterion.
	 * 
	 * @return the split criterion
	 */
	public abstract SplitCriterion<BinaryClassificationTarget> buildCriterion();
}
