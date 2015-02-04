package com.dsi.parallax.optimization.regularization;

import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.*;

import java.util.Arrays;

import static com.google.common.base.Preconditions.checkArgument;

public class TruncationConfigurableBuilder extends
		AbstractConfigurable<TruncationConfigurableBuilder> {

	private static final long serialVersionUID = -7467807933621577896L;
	private int period = 1;
	private TruncationType truncationType = TruncationType.NONE;
	private double alpha = 1d, theta = 1d;

	public static OptionSet<TruncationConfigurableBuilder> options = new TruncationBuilderOptions();

	public TruncationConfigurableBuilder() {
		this(new Configuration<TruncationConfigurableBuilder>(options));
	}

	public TruncationConfigurableBuilder(
			Configuration<TruncationConfigurableBuilder> config) {
		configure(config);
	}

	public static String optionInfo() {
		StringBuilder builder = new StringBuilder();
		for (Option option : options) {
			builder.append("-" + option.getShortName() + ": "
					+ option.getDescription() + ", ");
		}
		builder.replace(builder.length() - 2, builder.length(), "");
		return builder.toString();
	}

	/**
	 * change the period of gradient truncation updates. parameters are updated
	 * every p epochs
	 * 
	 * @param period
	 *            how often are parameters truncated? must be positive.
	 * @return
	 */
	public TruncationConfigurableBuilder setPeriod(int period) {
		checkArgument(period > 0, "period must be positive, given: %s", period);
		this.period = period;
		return this;
	}

	/**
	 * set the type of gradient truncation that is used.
	 * 
	 * @param truncationType
	 * @return
	 */
	public TruncationConfigurableBuilder setTruncationType(
			TruncationType truncationType) {
		this.truncationType = truncationType;
		return this;
	}

	/**
	 * parameter controlling the agressiveness of updates. higher alpha leads to
	 * a more aggressive reduction in parameter values
	 * 
	 * @param alpha
	 *            strength of parameter updates. must be non-negative
	 * @return
	 */
	public TruncationConfigurableBuilder setAlpha(double alpha) {
		checkArgument(alpha >= 0, "alpha must be non-negative, given: %s",
				alpha);
		this.alpha = alpha;
		return this;
	}

	/**
	 * threshold value, usually used for determining the minimum parameter value
	 * that will be truncated.
	 * 
	 * @param threshold
	 *            parameter threshold. must be non-negative
	 * @return
	 */
	public TruncationConfigurableBuilder setThreshold(double threshold) {
		checkArgument(threshold >= 0,
				"threshold must be non-negative, given: %s", threshold);
		this.theta = threshold;
		return this;
	}

	public int getPeriod() {
		return period;
	}

	public TruncationType getTruncationType() {
		return truncationType;
	}

	public double getAlpha() {
		return alpha;
	}

	public double getTheta() {
		return theta;
	}

	public GradientTruncation build() {
		switch (truncationType) {
		case ROUNDING:
			return new RoundingTruncation(period, theta);
		case TRUNCATING:
			return new TruncatedGradient(period, alpha, theta);
		case PEGASOS:
			return new PegasosTruncation(period, alpha);
		case MODDUCHI:
			return new ModifiedDuchiTruncation(period, alpha);
		case NONE:
		default:
			return new NullTruncation();
		}
	}

	@Override
	public Configuration<TruncationConfigurableBuilder> getConfiguration() {
		return new Configuration<TruncationConfigurableBuilder>(options);
	}

	@Override
	public Configuration<TruncationConfigurableBuilder> populateConfiguration(
			Configuration<TruncationConfigurableBuilder> conf) {
		conf.addEnumValueOnShortName("T", truncationType);
		conf.addIntegerValueOnShortName("p", period);
		conf.addFloatValueOnShortName("a", alpha);
		conf.addFloatValueOnShortName("t", theta);
		return conf;
	}

	@Override
	public void configure(
			Configuration<TruncationConfigurableBuilder> configuration) {
		setPeriod(configuration.integerOptionFromShortName("p"));
		setTruncationType((TruncationType) configuration.enumFromShortName("T"));
		setAlpha(configuration.floatOptionFromShortName("a"));
		setThreshold(configuration.floatOptionFromShortName("t"));
	}

	public static class TruncationBuilderOptions extends
			OptionSet<TruncationConfigurableBuilder> {
		{
			addOption(new IntegerOption(
					"p",
					"period",
					"period for gradient truncation- parameters are truncated every p epochs",
					1, false, new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(10000)));
			addOption(new EnumOption<TruncationType>("T", "truncationtype",
					false, "the type of truncation to use. options: "
							+ Arrays.toString(TruncationType.values()),
					TruncationType.class, TruncationType.NONE));
			addOption(new FloatOption("a", "alpha",
					"alpha governing the change in parameter values", 1., true,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(1000)));
			addOption(new FloatOption(
					"t",
					"threshold",
					"threshold for minimum parameter value needed for truncation; values less than thresh are truncated",
					1., true, new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
		}
	}

}
