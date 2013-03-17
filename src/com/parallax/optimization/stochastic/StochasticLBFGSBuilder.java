package com.parallax.optimization.stochastic;

import static com.google.common.base.Preconditions.checkArgument;

import com.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.parallax.ml.util.bounds.GreaterThanValueBound;
import com.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.FloatOption;
import com.parallax.ml.util.option.IntegerOption;
import com.parallax.optimization.stochastic.StochasticBFGSBuilder.StochasticBFGSBuilderOptions;

public class StochasticLBFGSBuilder extends
		StochasticGradientOptimizationBuilder<StochasticLBFGSBuilder> {

	private static final long serialVersionUID = 9222437112179674855L;
	private double lambda = 0.1, epsilon = Math.pow(10, -10);

	private int bandwidth = 50;

	public StochasticLBFGSBuilder(int dimension, boolean bias) {
		super(dimension, bias);
	}

	@Override
	protected StochasticLBFGSBuilder getThis() {
		return this;
	}

	@Override
	public StochasticLBFGS build() {
		return new StochasticLBFGS(dimension, bias,
				annealingScheduleConfigurableBuilder.build(),
				truncationBuilder.build(), buildCoefficientLossMap(),
				regularizeIntercept, regularizationWeight, bandwidth, epsilon,
				lambda);
	}

	public StochasticLBFGSBuilder setLambda(double lambda) {
		checkArgument(lambda >= 0, "lambda must be non-negative given: %s",
				lambda);
		this.lambda = lambda;
		return thisOptimizer;
	}

	public StochasticLBFGSBuilder setEpsilon(double lambda) {
		checkArgument(epsilon > 0, "epsilon must be greater than 0. given: %s",
				epsilon);
		this.lambda = lambda;
		return thisOptimizer;
	}

	public StochasticLBFGSBuilder setBandwidth(int bandwidth) {
		checkArgument(
				bandwidth > 0,
				"bandwidth, the number of prior gradients to consider, must be positive. given: %s",
				bandwidth);
		this.bandwidth = bandwidth;
		return thisOptimizer;
	}

	@Override
	public void configure(Configuration<StochasticLBFGSBuilder> conf) {
		super.configure(conf);
		setLambda(conf.floatOptionFromShortName("l"));
		setEpsilon(conf.floatOptionFromShortName("e"));
		setBandwidth(conf.integerOptionFromShortName("B"));
	}

	@Override
	public Configuration<StochasticLBFGSBuilder> getConfiguration() {
		Configuration<StochasticLBFGSBuilder> conf = new Configuration<StochasticLBFGSBuilder>(
				new StochasticLBFGSBuilderOptions());
		populateConfiguration(conf);
		return conf;
	}

	@Override
	public Configuration<StochasticLBFGSBuilder> populateConfiguration(
			Configuration<StochasticLBFGSBuilder> conf) {
		super.populateConfiguration(conf);
		conf.addFloatValueOnShortName("l", lambda);
		conf.addFloatValueOnShortName("e", epsilon);
		conf.addIntegerValueOnShortName("B", bandwidth);
		return conf;
	}

	public static StochasticBFGSBuilderOptions getOptions() {
		return new StochasticBFGSBuilderOptions();
	}

	public static class StochasticLBFGSBuilderOptions
			extends
			StochasticGradientOptimizationBuilderOptions<StochasticLBFGSBuilder> {
		{
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
			addOption(new IntegerOption("B", "bandwidth",
					"the number of prior gradients to consider", 50, false,
					new GreaterThanValueBound(0)));
		}

	}
}
