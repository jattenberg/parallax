package com.dsi.parallax.optimization.stochastic.anneal;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.OptionSet;

import static com.google.common.base.Preconditions.checkArgument;

public class AdaGradAnnealingScheduleBuilder extends AnnealingScheduleBuilder {

	private static final long serialVersionUID = -7374549992911987366L;

	public static final AdaGradAnnealingScheduleOptions options = new AdaGradAnnealingScheduleOptions();

	private double initialRate = 0.1;

	public AdaGradAnnealingScheduleBuilder(
			Configuration<AnnealingScheduleBuilder> config) {
		configure(config);
	}

	public AdaGradAnnealingScheduleBuilder() {
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
		conf.addFloatValueOnShortName("r", initialRate);
		return conf;
	}

	@Override
	public void configure(Configuration<AnnealingScheduleBuilder> configuration) {
		setInitialRate(configuration.floatOptionFromShortName("r"));
	}

	@Override
	public AdaGradAnnealingSchedule build() {
		return new AdaGradAnnealingSchedule(initialRate);
	}

	public AdaGradAnnealingScheduleBuilder setInitialRate(double initialRate) {
		checkArgument(initialRate > 0,
				"initial learning rate must be positive, given: %s",
				initialRate);
		this.initialRate = initialRate;
		return this;
	}

	public double getInitialLearningRate() {
		return initialRate;
	}

	@Override
	public Configuration<AnnealingScheduleBuilder> defaultConfiguration() {
		return new Configuration<AnnealingScheduleBuilder>(options);
	}

	@Override
	public AnnealingScheduleType correspondingType() {
		return AnnealingScheduleType.ADAGRAD;
	}

	public static class AdaGradAnnealingScheduleOptions extends
			OptionSet<AnnealingScheduleBuilder> {
		{
			addOption(new FloatOption(
					"r",
					"rate",
					"initial learning rate used for exponential annealing schedule",
					0.1, true, new GreaterThanValueBound(0)));
		}
	}
}
