/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ********************
import com.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableOption;

import com.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;
 **********************************************************/
package com.parallax.optimization.stochastic;

import java.util.Map;

import com.google.common.collect.Maps;
import com.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.parallax.ml.util.option.AbstractConfigurable;
import com.parallax.ml.util.option.BooleanOption;
import com.parallax.ml.util.option.ConfigurableOption;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.FloatOption;
import com.parallax.ml.util.option.OptionSet;
import com.parallax.optimization.regularization.LinearCoefficientLossType;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;
import com.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableOption;

public abstract class StochasticGradientOptimizationBuilder<O extends StochasticGradientOptimizationBuilder<O>>
		extends AbstractConfigurable<O> {

	private static final long serialVersionUID = 101320820924460220L;
	protected final int dimension;
	protected final boolean bias;

	protected AnnealingScheduleConfigurableBuilder annealingScheduleConfigurableBuilder = new AnnealingScheduleConfigurableBuilder();

	protected TruncationConfigurableBuilder truncationBuilder = new TruncationConfigurableBuilder();

	protected double gaussianWeight = 0, laplaceWeight = 0, cauchyWeight = 0,
			squaredWeight = 0, regularizationWeight = 1;

	protected boolean regularizeIntercept = false;

	protected final O thisOptimizer;

	/**
	 * @param dimension
	 * @param bias
	 */
	public StochasticGradientOptimizationBuilder(int dimension, boolean bias) {
		this.bias = bias;
		this.dimension = dimension + (bias ? 1 : 0);
		thisOptimizer = getThis();
	}

	public StochasticGradientOptimizationBuilder(int dimension, boolean bias,
			Configuration<O> config) {
		this(dimension, bias);
		configure(config);
	}

	protected abstract O getThis();

	public int getDimension() {
		return this.dimension - (bias ? 1 : 0);
	}

	public boolean getBias() {
		return this.bias;
	}

	public O setAnnealingScheduleConfigurableBuilder(
			AnnealingScheduleConfigurableBuilder annealingScheduleConfigurableBuilder) {
		this.annealingScheduleConfigurableBuilder = annealingScheduleConfigurableBuilder;
		return thisOptimizer;
	}

	public O configureAnnealingScheduleConfigurableBuilder(
			Configuration<AnnealingScheduleConfigurableBuilder> configuration) {
		this.annealingScheduleConfigurableBuilder.configure(configuration);
		return thisOptimizer;
	}

	public O setGradientTruncationBuilder(TruncationConfigurableBuilder builder) {
		this.truncationBuilder = builder;
		return thisOptimizer;
	}

	public O setGradientTruncationBuilder(
			Configuration<TruncationConfigurableBuilder> truncationConfig) {
		return setGradientTruncationBuilder(new TruncationConfigurableBuilder(
				truncationConfig));
	}

	public O setGaussianWeight(double gaussianWeight) {
		this.gaussianWeight = gaussianWeight;
		return thisOptimizer;
	}

	public O setLaplaceWeight(double laplaceWeight) {
		this.laplaceWeight = laplaceWeight;
		return thisOptimizer;
	}

	public O setCauchyWeight(double cauchyWeight) {
		this.cauchyWeight = cauchyWeight;
		return thisOptimizer;
	}

	public O setSquaredWeight(double squaredWeight) {
		this.squaredWeight = squaredWeight;
		return thisOptimizer;
	}

	public O setRegularizeIntercept(boolean regularizeIntercept) {
		this.regularizeIntercept = regularizeIntercept;
		return thisOptimizer;
	}

	public O setRegularizationWeight(double regularizationWeight) {
		this.regularizationWeight = regularizationWeight;
		return thisOptimizer;
	}

	protected Map<LinearCoefficientLossType, Double> buildCoefficientLossMap() {
		Map<LinearCoefficientLossType, Double> out = Maps.newHashMap();
		out.put(LinearCoefficientLossType.GAUSSIAN, gaussianWeight);
		out.put(LinearCoefficientLossType.LAPLACE, laplaceWeight);
		out.put(LinearCoefficientLossType.CAUCHY, cauchyWeight);
		out.put(LinearCoefficientLossType.SQUARED, squaredWeight);
		return out;
	}

	public abstract GradientStochasticOptimizer build();

	@Override
	public Configuration<O> getConfiguration() {
		Configuration<O> conf = new Configuration<O>(
				new StochasticGradientOptimizationBuilderOptions<O>());
		return populateConfiguration(conf);
	}

	@Override
	public void configure(Configuration<O> conf) {

		// set up gradient truncation
		@SuppressWarnings("unchecked")
		Configuration<TruncationConfigurableBuilder> truncConfig = (Configuration<TruncationConfigurableBuilder>) conf
				.configurationFromShortName("T");
		setGradientTruncationBuilder(truncConfig);

		// configure regularization
		setRegularizeIntercept(conf.booleanOptionFromShortName("i"));
		setRegularizationWeight(conf.floatOptionFromShortName("r"));

		setGaussianWeight(conf.floatOptionFromShortName("GR"));
		setLaplaceWeight(conf.floatOptionFromShortName("LR"));
		setCauchyWeight(conf.floatOptionFromShortName("CR"));
		setSquaredWeight(conf.floatOptionFromShortName("SR"));

		@SuppressWarnings("unchecked")
		Configuration<AnnealingScheduleConfigurableBuilder> annealConfig = (Configuration<AnnealingScheduleConfigurableBuilder>) conf
				.nestedConfigurationFromShortName("a");

		configureAnnealingScheduleConfigurableBuilder(annealConfig);

	}

	@Override
	public Configuration<O> populateConfiguration(Configuration<O> conf) {
		conf.addConfigurableValueOnShortName("T",
				truncationBuilder.getConfiguration());

		conf.addFloatValueOnShortName("r", regularizationWeight);

		conf.addFloatValueOnShortName("GR", gaussianWeight);
		conf.addFloatValueOnShortName("LR", laplaceWeight);
		conf.addFloatValueOnShortName("CR", cauchyWeight);
		conf.addFloatValueOnShortName("SR", squaredWeight);

		conf.addNestedConfigurableValueOnShortName("a",
				annealingScheduleConfigurableBuilder.getConfiguration());

		return conf;
	}

	protected static class StochasticGradientOptimizationBuilderOptions<O extends StochasticGradientOptimizationBuilder<O>>
			extends OptionSet<O> {
		{
			addOption(new BooleanOption("i", "regularizeintercept",
					"apply regularization to the intercept term", false, true));
			addOption(new ConfigurableOption<TruncationConfigurableBuilder>(
					"T", "truncationConfig", false,
					"the configuration for builders of gradient truncations. Options: "
							+ TruncationConfigurableBuilder.optionInfo(),
					new Configuration<TruncationConfigurableBuilder>(
							TruncationConfigurableBuilder.options)));

			// regularization coefficients
			addOption(new FloatOption("GR", "gaussianWeight",
					"weight on gaussian regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("LR", "laplaceWeight",
					"weight on laplacian regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("CR", "cauchyWeight",
					"weight on cauchian regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("SR", "squaredWeight",
					"weight on squared loss regularization", 0, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("r", "regweight",
					"weight on regularization", 1, false,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new AnnealingScheduleConfigurableOption("a", "anneal",
					true,
					"configurable option for the type of annealing schedule to use"));
		}
	}
}
