/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.regularization;

import com.dsi.parallax.ml.util.MLUtils;

/**
 * class for gradients of the regularization components of regularized linear
 * functions combinations may be maintained by utilizing a enumset in the client
 * class.
 * 
 * @author josh
 * 
 */
public enum LinearCoefficientLossType {

	UNIFORM {
		@Override
		public double gradientNonTruncated(double paramValue, double variance) {
			return 0;
		}
	},
	GAUSSIAN {
		@Override
		public double gradientNonTruncated(double paramValue, double variance) {
			return -paramValue / Math.pow(variance, 2);
		}
	},
	LAPLACE {
		@Override
		public double gradientNonTruncated(double paramValue, double variance) {
			return -MLUtils.ROOT2 * Math.signum(paramValue) / variance;
		}
	},
	CAUCHY {
		@Override
		public double gradientNonTruncated(double paramValue, double variance) {
			return -2. * paramValue
					/ (Math.pow(paramValue, 2) + Math.pow(variance, 2));
		}
	},
	SQUARED {
		@Override
		public double gradientNonTruncated(double paramValue, double variance) {
			return -paramValue;
		}
	};

	public abstract double gradientNonTruncated(double paramValue,
			double variance);

	/**
	 * ensures that the regularization doesnt "pass zero"
	 * 
	 * @param paramValue
	 *            value of the param being shrunk
	 * @param variance
	 *            additional regularizaiton param
	 * @return
	 */
	public double gradient(double paramValue, double variance) {
        double loss = gradientNonTruncated(paramValue, variance);
		if (Math.signum(paramValue) * Math.signum(paramValue - loss) < 0d) {
            loss = Math.signum(loss) * paramValue;
        }
		return loss;
	}
}
