/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.bayes;

import com.dsi.parallax.ml.classifier.AbstractUpdateableClassifier;
import com.dsi.parallax.ml.distributions.Distribution;
import com.dsi.parallax.ml.distributions.DistributionConfigurableBuilder;
import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.option.Configuration;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/**
 * binary naive bayes using a variety of density estimators
 * 
 * TODO: better construction and handling of kernel options.
 * 
 * @author jattenberg
 */
public class NaiveBayes extends AbstractUpdateableClassifier<NaiveBayes> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -8875379865195128124L;

	/** The Constant CLASSES. */
	private static final int POS = 1, CLASSES = 2; // class indicies

	/** The prior. */
	private Distribution prior;

	/** The likelihoods. */
	private Distribution[] likelihoods;

	/** The document length normalization. */
	private double documentLengthNormalization = 100; // no document
														// normalization by
														// default
	/** The distribution builder. */
	private DistributionConfigurableBuilder distributionBuilder = new DistributionConfigurableBuilder();

	/** The prior distribution builder. */
	private DistributionConfigurableBuilder priorDistributionBuilder = new DistributionConfigurableBuilder();

	/**
	 * Instantiates a new naive bayes.
	 * 
	 * @param dimension
	 *            the dimension
	 * @param bias
	 *            the bias
	 */
	public NaiveBayes(int dimension, boolean bias) {
		super(dimension, bias);
		passes = 1;
		initialize();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#update(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	public <I extends Instance<BinaryClassificationTarget>> void updateModel(I x) {
		BinaryClassificationTarget label = x.getLabel();
		double oneNorm = x.L1Norm();

		if (label == null || oneNorm <= 0 || x.getWeight() <= 0)
			return;
		double weight = x.getWeight()
				* (documentLengthNormalization > 0 ? documentLengthNormalization
						/ oneNorm
						: 1.);
		double[] labels = new double[] { 1. - label.getValue(),
				label.getValue() };

		for (int lab = 0; lab < CLASSES; lab++) {
			double labelWeight = labels[lab];
			for (int x_i : x) {
				double y_i = x.getFeatureValue(x_i);
				likelihoods[lab].observe(x_i, y_i * labelWeight * weight);
			}
			prior.observe(lab, labelWeight * weight);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.AbstractClassifier#regress(com.parallax.ml
	 * .instance.Instanze)
	 */
	@Override
	protected double regress(Instance<?> x) {
		double[] scores = prior.logProbabilities();

		for (int cat = 0; cat < CLASSES; cat++) {
			for (int x_i : x) {
				double y_i = x.getFeatureValue(x_i);
				scores[cat] += likelihoods[cat].logProbability(x_i, y_i);
			}
		}

		double maxScore = MLUtils.max(scores);
		for (int cat = 0; cat < CLASSES; cat++)
			scores[cat] -= maxScore;

		// Exponentiate and normalize
		double sum = 0;
		for (int cat = 0; cat < CLASSES; cat++)
			sum += (scores[cat] = Math.exp(scores[cat]));
		for (int cat = 0; cat < CLASSES; cat++)
			scores[cat] /= sum;
		return scores[POS];
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.Model#initialize()
	 */
	@Override
	public NaiveBayes initialize() {
		passes = 1;
		prior = priorDistributionBuilder.build(CLASSES);
		likelihoods = new Distribution[CLASSES];
		for (int cat = 0; cat < CLASSES; cat++)
			likelihoods[cat] = distributionBuilder.build(dimension);
		return model;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.model.AbstractModel#getModel()
	 */
	@Override
	protected NaiveBayes getModel() {
		return this;
	}

	/**
	 * Gets the doc length normalization.
	 * 
	 * @return the doc length normalization
	 */
	public double getDocLengthNormalization() {
		return documentLengthNormalization;
	}

	/**
	 * Gets the distribution builder.
	 * 
	 * @return the distribution builder
	 */
	public DistributionConfigurableBuilder getDistributionBuilder() {
		return distributionBuilder;
	}

	/**
	 * Gets the prior distribution builder.
	 * 
	 * @return the prior distribution builder
	 */
	public DistributionConfigurableBuilder getPriorDistributionBuilder() {
		return priorDistributionBuilder;
	}

	/**
	 * Sets the document length normalization.
	 * 
	 * @param docLengthNormalization
	 *            the doc length normalization
	 * @return the naive bayes
	 */
	public NaiveBayes setDocumentLengthNormalization(
			double docLengthNormalization) {
		checkArgument(docLengthNormalization > 0,
				"docLengthNormalization must be positive. input: %s",
				docLengthNormalization);
		this.documentLengthNormalization = docLengthNormalization;
		return model;
	}

	/**
	 * set the builder for liklihood distributions used in the naive bayes
	 * model.
	 * 
	 * @param distributionBuilder
	 *            builder for distirbutions
	 * @return the naive bayes
	 */
	public NaiveBayes setDistributionBuilder(
			DistributionConfigurableBuilder distributionBuilder) {
		this.distributionBuilder = distributionBuilder;
		return model;
	}

	/**
	 * set the builder for likelihood distributions used in the naive bayes
	 * model using a configuraion.
	 * 
	 * @param distributionBuilderConfiguration
	 *            configuration for building a distribuiton builder
	 * @return the naive bayes
	 */
	public NaiveBayes setDistributionBuilder(
			Configuration<DistributionConfigurableBuilder> distributionBuilderConfiguration) {
		return setDistributionBuilder(new DistributionConfigurableBuilder(
				distributionBuilderConfiguration));
	}

	/**
	 * set the builder for prior distributions used in the naive bayes model.
	 * 
	 * @param priorDistributionBuilder
	 *            the prior distribution builder
	 * @return the naive bayes
	 */
	public NaiveBayes setPriorDistributionBuilder(
			DistributionConfigurableBuilder priorDistributionBuilder) {
		this.priorDistributionBuilder = priorDistributionBuilder;
		return model;
	}

	/**
	 * set the builder for prior distributions used in the naive bayes model
	 * using a configuraion.
	 * 
	 * @param priorDistributionBuilderConfiguration
	 *            the prior distribution builder configuration
	 * @return the naive bayes
	 */
	public NaiveBayes setPriorDistributionBuilder(
			Configuration<DistributionConfigurableBuilder> priorDistributionBuilderConfiguration) {
		return setPriorDistributionBuilder(new DistributionConfigurableBuilder(
				priorDistributionBuilderConfiguration));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.classifier.UpdateableClassifier#updateModel(com.parallax.ml
	 * .instance.Instances)
	 */
	@Override
	public <I extends Instances<? extends Instance<BinaryClassificationTarget>>> void updateModel(
			I instst) {
		for (Instance<BinaryClassificationTarget> inst : instst) {
			update(inst);
		}

	}

}
