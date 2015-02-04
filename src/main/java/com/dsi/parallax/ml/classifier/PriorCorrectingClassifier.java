/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.OptionSet;

import java.io.Serializable;

import static com.google.common.base.Preconditions.checkArgument;
//TODO: revise to make more configurable, remove final fields. 
/**
 * decorator for a correcting the probability estimates of a classifier trained
 * on base rate q making predictions on base rate p. See
 * {@link <a href="http://blog.smola.org/post/4110255196/real-simple-covariate-shift-correction">Real simple covariate shift correction</a>}
 * for more info
 * 
 * 
 * 
 * @author josh
 */
public class PriorCorrectingClassifier implements
		Classifier<PriorCorrectingClassifier>, Serializable {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -5133832270203617542L;

	/** The model where probabilities are being corrected */
	private final Classifier<?> model;

	/** The probability of positive examples in the training set */
	private final double trainingProb;
	/** The probability of positive examples in the test set */
	private final double evaluationProb;

	/**
	 * Instantiates a new prior correcting classifier.
	 * 
	 * @param c
	 *            the c
	 * @param trainingProb
	 *            the probability of positive examples in the training set
	 * @param evaluationProb
	 *            the probability of positive examples in the test set
	 */
	public PriorCorrectingClassifier(Classifier<?> c, double trainingProb,
			double evaluationProb) {
		this.model = c;
		this.trainingProb = trainingProb;
		this.evaluationProb = evaluationProb;
		checkArgument(trainingProb >= 0 && trainingProb <= 1,
				"trainingProb must be >= 0 & <= 1 input: %s", trainingProb);
		checkArgument(evaluationProb >= 0 && evaluationProb <= 1,
				"evaluationProb must be >= 0 & <= 1 input: %s", evaluationProb);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#getModelDimension()
	 */
	@Override
	public int getModelDimension() {
		return this.model.getModelDimension();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public PriorCorrectingClassifier initialize() {
		throw new UnsupportedOperationException(
				"prior correcting classifier can't be initialized");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#usesBiasTerm()
	 */
	@Override
	public boolean usesBiasTerm() {
		throw new UnsupportedOperationException(
				"prior correcting classifier can't be initialized");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#predict(com.parallax.ml.instance.Instanze)
	 */
	@Override
	public BinaryClassificationTarget predict(Instance<?> instance) {
		BinaryClassificationTarget target = model.predict(instance);
		target.setValue(target.getValue() * evaluationProb / trainingProb);
		return target;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.model.Model#train(com.parallax.ml.instance.Instances)
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void train(
			I instances) {
		throw new UnsupportedOperationException(
				"prior correcting classifier can't be initialized");
	}

	/**
	 * Gets the probability of positive examples in the training set
	 * 
	 * @return the probability of positive examples in the training set
	 */
	public double getTrainingProb() {
		return trainingProb;
	}

	/**
	 * Gets the probability of positive examples in the test set
	 * 
	 * @return theprobability of positive examples in the test set
	 */
	public double getEvaluationProb() {
		return evaluationProb;
	}

	/**
	 * A simple builder for {@link PriorCorrectingClassifier}
	 * largely implemented for completeness
	 */
	public static class PriorCorrectingClassifierBuilder
			extends
			ClassifierBuilder<PriorCorrectingClassifier, PriorCorrectingClassifierBuilder> {

		/** The Constant serialVersionUID. */
		private static final long serialVersionUID = -1302007257559136776L;

		/** The options. */
		public static OptionSet<PriorCorrectingClassifierBuilder> options = new PriorCorrectingClassifierOptions();

		/** The evaluation prob. */
		private double trainingProb = 0.5; 
		private double evaluationProb = 0.5;

		/**
		 * Instantiates a new prior correcting classifier builder.
		 * 
		 * @param config
		 *            the config
		 */
		public PriorCorrectingClassifierBuilder(
				Configuration<PriorCorrectingClassifierBuilder> config) {
			super(config);
			configure(config);
		}

		/**
		 * Instantiates a new prior correcting classifier builder.
		 * 
		 * @param dimension
		 *            the dimension
		 * @param bias
		 *            the bias
		 */
		public PriorCorrectingClassifierBuilder(int dimension, boolean bias) {
			super(dimension, bias);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#build()
		 */
		@Override
		public PriorCorrectingClassifier build() {
			// TODO Auto-generated method stub
			return null;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.model.ModelBuilder#getThis()
		 */
		@Override
		protected PriorCorrectingClassifierBuilder getThis() {
			return this;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see com.parallax.ml.classifier.ClassifierBuilder#getConfiguration()
		 */
		@Override
		public Configuration<PriorCorrectingClassifierBuilder> getConfiguration() {
			Configuration<PriorCorrectingClassifierBuilder> conf = new Configuration<PriorCorrectingClassifierBuilder>(
					options);
			return populateConfiguration(conf);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * com.parallax.ml.classifier.ClassifierBuilder#populateConfiguration
		 * (com.parallax.ml.util.option.Configuration)
		 */
		@Override
		public Configuration<PriorCorrectingClassifierBuilder> populateConfiguration(
				Configuration<PriorCorrectingClassifierBuilder> conf) {
			conf.addFloatValueOnShortName("T", trainingProb);
			conf.addFloatValueOnShortName("E", evaluationProb);
			return conf;
		}

		/**
		 * The Options for prior correcting classifier Builders. <br>
		 * options are:
		 * T / trainingprob [0, 1] - The probability of positive examples in the training set <br>
		 * E / evalprob [0,1] - The probability of positive examples in the test set
		 * 
		 */
		public static class PriorCorrectingClassifierOptions extends
				OptionSet<PriorCorrectingClassifierBuilder> {
			{
				addOption(new FloatOption("T", "trainingprob",
						"probability of positive examples in the training set",
						0, false, new GreaterThanOrEqualsValueBound(0),
						new LessThanOrEqualsValueBound(1)));
				addOption(new FloatOption(
						"E",
						"evalprob",
						"probability of positive examples in the evaluation set",
						0, false, new GreaterThanOrEqualsValueBound(0),
						new LessThanOrEqualsValueBound(1)));
			}
		}
	}

}
