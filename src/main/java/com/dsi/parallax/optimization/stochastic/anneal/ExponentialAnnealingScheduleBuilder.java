package com.dsi.parallax.optimization.stochastic.anneal;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.OptionSet;

public class ExponentialAnnealingScheduleBuilder extends
		AnnealingScheduleBuilder {

	private static final long serialVersionUID = 255090029367623565L;

	public static final ExponentialAnnealingScheduleOptions options = new ExponentialAnnealingScheduleOptions();

	private double initialLearningRate = 0.1;
	private double base = 0.2;

	public ExponentialAnnealingScheduleBuilder(
			Configuration<AnnealingScheduleBuilder> configuration) {
		configure(configuration);
	}

	public ExponentialAnnealingScheduleBuilder() {
		this(new Configuration<AnnealingScheduleBuilder>(options));
	}

	@Override
	public Configuration<AnnealingScheduleBuilder> getConfiguration() {
		Configuration<AnnealingScheduleBuilder> conf = new Configuration<AnnealingScheduleBuilder>(
				options);
		populateConfiguration(conf);
		return conf;
	}

	@Override
	public Configuration<AnnealingScheduleBuilder> populateConfiguration(
			Configuration<AnnealingScheduleBuilder> conf) {
		conf.addFloatValueOnShortName("r", initialLearningRate);
		conf.addFloatValueOnShortName("b", base);
		return conf;
	}

	@Override
	public void configure(Configuration<AnnealingScheduleBuilder> configuration) {
		setInitialRate(configuration.floatOptionFromShortName("r"));
		setExponentialBase(configuration.floatOptionFromShortName("b"));
	}

	@Override
	public ExponentialAnnealingSchedule build() {
		return new ExponentialAnnealingSchedule(initialLearningRate, base);
	}

	public ExponentialAnnealingScheduleBuilder setInitialRate(double initialRate) {
		checkArgument(initialRate > 0,
				"initial learning rate must be positive, given: %s",
				initialRate);
		this.initialLearningRate = initialRate;
		return this;
	}

	public ExponentialAnnealingScheduleBuilder setExponentialBase(
			double exponentialBase) {
		checkArgument(exponentialBase > 0 && exponentialBase <= 1,
				"exponential base must be in (0, 1], given: %s",
				exponentialBase);
		this.base = exponentialBase;
		return this;
	}

	public double getInitialLearningRate() {
		return initialLearningRate;
	}

	public double getExponentialBase() {
		return base;
	}

	public static class ExponentialAnnealingScheduleOptions extends
			OptionSet<AnnealingScheduleBuilder> {
		{
			addOption(new FloatOption(
					"r",
					"rate",
					"initial learning rate used for exponential annealing schedule",
					0.1, true, new GreaterThanValueBound(0)));
			addOption(new FloatOption("b", "base",
					"base in exponential learning rate, must be in (0, 1]",
					0.2, true, new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(1)));
		}
	}

	@Override
	public Configuration<AnnealingScheduleBuilder> defaultConfiguration() {
		return new Configuration<AnnealingScheduleBuilder>(options);
	}

	@Override
	public AnnealingScheduleType correspondingType() {
		return AnnealingScheduleType.EXPONENTIAL;
	}
}
