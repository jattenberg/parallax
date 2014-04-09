/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.evaluation;

// TODO: Auto-generated Javadoc
/**
 * The Enum LossGradientType.
 */
public enum LossGradientType {
	
	/** The hingeloss. */
	HINGELOSS {
		@Override
		public double computeLossUpdate(double prediction, double y) {
			double z = prediction * y;
			if (z <= 1.)
				return y;
			return 0.0;
		}
	},
	
	/** The logloss. */
	LOGLOSS {
		@Override
		public double computeLossUpdate(double prediction, double y) {
			double z = prediction * y;
			// approximately equal and saves the computation of the log
			if (z > 18.0)
				return Math.exp(-z) * y;
			if (z < -18.0)
				return y;
			return y / (Math.exp(z) + 1.0);
		}
	},
	
	/** The squaredloss. */
	SQUAREDLOSS {
		@Override
		public double computeLossUpdate(double prediction, double y) {
			return -prediction + y;
		}
	};
	
	/**
	 * Compute loss update.
	 *
	 * @param prediction the prediction
	 * @param y the y
	 * @return the double
	 */
	public abstract double computeLossUpdate(double prediction, double y);
}
