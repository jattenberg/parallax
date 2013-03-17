/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.stochastic;

import com.parallax.ml.util.option.Configuration;

public class SGDBuilder extends
		StochasticGradientOptimizationBuilder<SGDBuilder> {

	private static final long serialVersionUID = 5721349133681420291L;

	public SGDBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	public StochasaticGradientDescent build() {
		return new StochasaticGradientDescent(dimension, bias,
				annealingScheduleConfigurableBuilder.build(), truncationBuilder.build(),
				buildCoefficientLossMap(), regularizeIntercept,
				regularizationWeight);
	}

	@Override
	protected SGDBuilder getThis() {
		return this;
	}

	@Override
	public void configure(Configuration<SGDBuilder> conf) {
		super.configure(conf);
	}

	@Override
	public Configuration<SGDBuilder> getConfiguration() {
		Configuration<SGDBuilder> conf = new Configuration<SGDBuilder>(
				new SGDOptions());
		populateConfiguration(conf);
		return conf;
	}

	@Override
	public Configuration<SGDBuilder> populateConfiguration(
			Configuration<SGDBuilder> conf) {
		super.populateConfiguration(conf);
		return conf;
	}

	public static SGDOptions getOptions() {
		return new SGDOptions();
	}

	protected static class SGDOptions extends
			StochasticGradientOptimizationBuilderOptions<SGDBuilder> {
		{
			//nothing here. 
		}

	}

}
