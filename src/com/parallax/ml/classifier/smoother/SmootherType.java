/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.smoother;

import java.util.Collection;

import com.parallax.ml.util.pair.PrimitivePair;

/**
 * an enumeration over different types of smoothers each enumeration defines
 * a factory method to the referenced smoothing technique either providing
 * an untrained model, or a trained model given appropriate predictions / labels
 * from a classifier.
 * 
 * @author josh
 */
public enum SmootherType {

	/**
	 * platt's smoothing performed with an updateable (SGD) MLE logistic
	 * regression extends the AbstractUpdateableSmoother class, implementing
	 * UpdateableSmoother.
	 */
	UPDATEABLEPLATT {
		@Override
		public Smoother<?> getSmoother() {
			return new UpdateableLogisticRegressionSmoother();
		}
	},
	/**
	 * platt's smoothing backed by the LogisticRegressionSmoother trained in
	 * batch mode.
	 */
	PLATT {
		@Override
		public Smoother<?> getSmoother() {
			return new LogisticRegressionSmoother();
		}

	},
	/**
	 * regularization with isotonic regression
	 */
	ISOTONIC {
		@Override
		public Smoother<?> getSmoother() {
			return new IsotonicSmoother();
		}
	},
	/**
	 * use the simple binning Smoother
	 */
	BINNING {

		@Override
		public Smoother<?> getSmoother() {
			return new BinningSmoother();
		}

	},
	/**
	 * simply applies a logistic function to a model's raw predictive output:
	 * 1/(1+exp(-pred))
	 */
	LOGIT {

		@Override
		public Smoother<?> getSmoother() {
			return new LogitSmoother();
		}

	},
	/**
	 * regularization using K-NN
	 */
	KNN {
		@Override
		public Smoother<?> getSmoother() {
			return new KNNSmoother();
		}
	},
	/**
	 * regularization using a local logistic regression
	 */
	LOCALLR {
		@Override
		public Smoother<?> getSmoother() {
			return new LocalLogisticRegressionSmoother();
		}
	},
	/** simply bounds the input score to the range [0,1]. */
	NONE {
		@Override
		public Smoother<?> getSmoother() {
			return new NullSmoother();
		}

	};

	/**
	 * method for returning an untrained smoother of the specified type.
	 * 
	 * @return Smoother
	 */
	public abstract Smoother<?> getSmoother();

	/**
	 * method for initializing and training a probability smoother.
	 * 
	 * @param input
	 *            List<Pair>, each pair contains a label and a prediction used
	 *            for smoothing
	 * @return Smoother trained smoother on the input data
	 */
	public Smoother<?> getAndTrainSmoother(Collection<PrimitivePair> input) {
		Smoother<?> r = getSmoother();
		r.train(input);
		return r;
	}
}
