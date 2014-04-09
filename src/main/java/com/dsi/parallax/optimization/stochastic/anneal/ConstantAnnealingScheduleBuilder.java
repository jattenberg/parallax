package com.dsi.parallax.optimization.stochastic.anneal;

import static com.google.common.base.Preconditions.checkArgument;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.OptionSet;

public class ConstantAnnealingScheduleBuilder extends AnnealingScheduleBuilder {

	private static final long serialVersionUID = -1693978478292653760L;

	public static OptionSet<AnnealingScheduleBuilder> options = new ConstantAnnealingScheduleOptions();

	private double learningRate = 0.1;

	public ConstantAnnealingScheduleBuilder() {
		this(new Configuration<AnnealingScheduleBuilder>(options));
	}

	public ConstantAnnealingScheduleBuilder(
			Configuration<AnnealingScheduleBuilder> config) {
		configure(config);
	}

	@Override
	public AnnealingScheduleType correspondingType() {
		return AnnealingScheduleType.CONSTANT;
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
		conf.addFloatValueOnShortName("r", getLearningRate());
		return conf;
	}

	@Override
	public void configure(Configuration<AnnealingScheduleBuilder> configuration) {
		setInitialRate(configuration.floatOptionFromShortName("r"));
	}

	public ConstantAnnealingScheduleBuilder setInitialRate(double learningRate) {
		checkArgument(learningRate > 0,
				"learning rate must be positive, given: %s", learningRate);
		this.learningRate = learningRate;
		return this;
	}

	public double getLearningRate() {
		return learningRate;
	}

	@Override
	public ConstantAnnealingSchedule build() {
		return new ConstantAnnealingSchedule(learningRate);
	}

	public static class ConstantAnnealingScheduleOptions extends
			OptionSet<AnnealingScheduleBuilder> {
		{
			addOption(new FloatOption("r", "rate",
					"learning rate used for constant annealing schedule", 0.1,
					true, new GreaterThanValueBound(0)));
		}
	}

	@Override
	public Configuration<AnnealingScheduleBuilder> defaultConfiguration() {
		return new Configuration<AnnealingScheduleBuilder>(options);
	}
}
