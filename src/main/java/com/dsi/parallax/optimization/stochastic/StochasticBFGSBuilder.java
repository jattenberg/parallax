/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;

import static com.google.common.base.Preconditions.checkArgument;

public class StochasticBFGSBuilder extends
		StochasticGradientOptimizationBuilder<StochasticBFGSBuilder> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7530427584695943782L;
	private double c = 0.9, lambda = 0.1, epsilon = Math.pow(10, -10);

	public StochasticBFGSBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	public StochasticBFGSBuilder setC(double c) {
		checkArgument(c > 0 && c <= 1, "c must be in (0, 1]. given: %s", c);
		this.c = c;
		return thisOptimizer;
	}

	public StochasticBFGSBuilder setLambda(double lambda) {
		checkArgument(lambda >= 0, "lambda must be non-negative given: %s",
				lambda);
		this.lambda = lambda;
		return thisOptimizer;
	}

	public StochasticBFGSBuilder setEpsilon(double lambda) {
		checkArgument(epsilon > 0, "epsilon must be greater than 0. given: %s",
				epsilon);
		this.lambda = lambda;
		return thisOptimizer;
	}

	@Override
	public StochasticBFGS build() {
		return new StochasticBFGS(dimension, bias,
				annealingScheduleConfigurableBuilder.build(),
				truncationBuilder.build(), buildCoefficientLossMap(),
				regularizeIntercept, regularizationWeight, epsilon, c, lambda);
	}

	@Override
	protected StochasticBFGSBuilder getThis() {
		return this;
	}

	@Override
	public void configure(Configuration<StochasticBFGSBuilder> conf) {
		super.configure(conf);
		setC(conf.floatOptionFromShortName("c"));
		setLambda(conf.floatOptionFromShortName("l"));
		setEpsilon(conf.floatOptionFromShortName("e"));
	}

	@Override
	public Configuration<StochasticBFGSBuilder> getConfiguration() {
		Configuration<StochasticBFGSBuilder> conf = new Configuration<StochasticBFGSBuilder>(
				new StochasticBFGSBuilderOptions());
		populateConfiguration(conf);
		return conf;
	}

	@Override
	public Configuration<StochasticBFGSBuilder> populateConfiguration(
			Configuration<StochasticBFGSBuilder> conf) {
		super.populateConfiguration(conf);
		conf.addFloatValueOnShortName("c", c);
		conf.addFloatValueOnShortName("l", lambda);
		conf.addFloatValueOnShortName("e", epsilon);
		return conf;
	}

	public static StochasticBFGSBuilderOptions getOptions() {
		return new StochasticBFGSBuilderOptions();
	}

	public static class StochasticBFGSBuilderOptions extends
			StochasticGradientOptimizationBuilderOptions<StochasticBFGSBuilder> {
		{
			addOption(new FloatOption("c", "c",
					"c parameter in sBFGS controlling update damping", 0.1,
					false, new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(1)));
			addOption(new FloatOption("l", "lambda",
					"lambda for sBFGS controlling impact of prior gradients",
					0.1, false, new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption(
					"e",
					"epsilon",
					"epsilon for sBFGS, initial value on diag of hessian inverse",
					Math.pow(10, -10), false, new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
		}

	}
}
