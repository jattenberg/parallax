package com.dsi.parallax.optimization.stochastic.anneal;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.OptionSet;

public class InverseDecayAnnealingScheduleBuilder extends
		AnnealingScheduleBuilder {

	private static final long serialVersionUID = 6674230108174948979L;

	public static final InverseDecayAnnealingScheduleOptions options = new InverseDecayAnnealingScheduleOptions();

	private double initialLearningRate = 0.1;
	private double decay = 0.1;

	public InverseDecayAnnealingScheduleBuilder(
			Configuration<AnnealingScheduleBuilder> configuration) {
		configure(configuration);
	}

	public InverseDecayAnnealingScheduleBuilder() {
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
		conf.addFloatValueOnShortName("d", decay);
		return conf;
	}

	@Override
	public void configure(Configuration<AnnealingScheduleBuilder> configuration) {
		setInitialRate(configuration.floatOptionFromShortName("r"));
		setDecay(configuration.floatOptionFromShortName("d"));
	}

	@Override
	public InverseDecayAnnealingSchedule build() {
		return new InverseDecayAnnealingSchedule(initialLearningRate, decay);
	}

	public InverseDecayAnnealingScheduleBuilder setInitialRate(
			double initialRate) {
		checkArgument(initialRate > 0,
				"initial learning rate must be positive, given: %s",
				initialRate);
		this.initialLearningRate = initialRate;
		return this;
	}

	public InverseDecayAnnealingScheduleBuilder setDecay(double decay) {
		checkArgument(decay > 0, "decay must be non-negative, given: %s", decay);
		this.decay = decay;
		return this;
	}

	public double getInitialLearningRate() {
		return initialLearningRate;
	}

	public double getDecay() {
		return decay;
	}
	
	@Override
	public Configuration<AnnealingScheduleBuilder> defaultConfiguration() {
		return new Configuration<AnnealingScheduleBuilder>(options);
	}

	@Override
	public AnnealingScheduleType correspondingType() {
		return AnnealingScheduleType.INVERSE;
	}

	public static class InverseDecayAnnealingScheduleOptions extends
			OptionSet<AnnealingScheduleBuilder> {
		{
			addOption(new FloatOption(
					"r",
					"rate",
					"initial learning rate used for exponential annealing schedule",
					0.1, true, new GreaterThanValueBound(0)));
			addOption(new FloatOption(
					"d",
					"decay",
					"decay factor, controls stepsize like: r/(1 + epoch/decay). Must be > 0",
					0.2, true, new GreaterThanValueBound(0)));
		}
	}

}
