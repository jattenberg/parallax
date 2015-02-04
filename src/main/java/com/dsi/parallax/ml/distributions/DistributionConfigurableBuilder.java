package com.dsi.parallax.ml.distributions;

import com.dsi.parallax.ml.distributions.kde.KDEConfigurableBuilder;
import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.*;

import java.util.Arrays;

/**
 * A configurable builder for multivariate probability distributions
 * 
 * @author jattenberg
 */
public class DistributionConfigurableBuilder extends
		AbstractConfigurable<DistributionConfigurableBuilder> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = -3539685135343543864L;

	/** The type of distribution to be built. */
	private DistributionType distType;

	/** Alpha used for laplace smoothing */
	private double alpha = 0;

	/** bins used for histogram distributions */
	private int bins = 100;

	/** The kde builder used for kde distributions. */
	private KDEConfigurableBuilder kdeBuilder = new KDEConfigurableBuilder();

	/**
	 * The options used for configuring {@link DistributionConfigurableBuilder}
	 * s.
	 */
	public static OptionSet<DistributionConfigurableBuilder> options = new DistributionConfigurableOptions();

	/**
	 * Instantiates a new distribution configurable builder.
	 */
	public DistributionConfigurableBuilder() {
		this(new Configuration<DistributionConfigurableBuilder>(options));
	}

	/**
	 * Instantiates a new distribution configurable builder.
	 * 
	 * @param config
	 *            containing the parameters describing the desired distributions
	 */
	public DistributionConfigurableBuilder(
			Configuration<DistributionConfigurableBuilder> config) {
		configure(config);
	}

	/**
	 * Builds a multivariate distribution with the desired number of covariates.
	 * 
	 * @param dimension
	 *            the number of covariates in the desired distribution
	 * @return the distribution
	 */
	public Distribution build(int dimension) {
		switch (distType) {
		case BERNOULLI:
			return new BernoulliMultivariateDistribution(dimension, alpha);
		case GAUSSIAN:
			return new GaussianDistribution(dimension);
		case HISTOGRAM:
			return new HistogramDistribution(dimension, bins);
		case KDE:
			return kdeBuilder.build(dimension);
		case MULTINOMIAL:
		default:
			return new MultinomialDistribution(dimension, alpha);
		}
	}

	/**
	 * the type of distribution that distribution builder will create.
	 * 
	 * @param distType
	 *            the desired distribution type
	 * @return the distribution configurable builder used for method chaining
	 */
	public DistributionConfigurableBuilder setDistributionType(
			DistributionType distType) {
		this.distType = distType;
		return this;
	}

	/**
	 * Gets the distribution type.
	 * 
	 * @return the distribution type
	 */
	public DistributionType getDistributionType() {
		return distType;
	}

	/**
	 * set alpha- the laplace-smoothing beta binomial/multinomial parameter used
	 * bernoulli / multinomial distributions: (alpha + successes) / (2 * alpha +
	 * trials);.
	 * 
	 * @param alpha
	 *            the alpha used for lapace smoothing
	 * @return the distribution configurable builder used for method chaining
	 */
	public DistributionConfigurableBuilder setAlpha(double alpha) {
		this.alpha = alpha;
		return this;
	}

	/**
	 * Gets the alpha.
	 * 
	 * @return the alpha
	 */
	public double getAlpha() {
		return alpha;
	}

	/**
	 * the number of bins in histogram distributions.
	 * 
	 * @param bins
	 *            the bins used for histogram distributions
	 * @return the distribution configurable builder used for method chaining
	 */
	public DistributionConfigurableBuilder setBins(int bins) {
		this.bins = bins;
		return this;
	}

	/**
	 * Sets the kde builder, presumably this builder has already been configured
	 * in the desired way
	 * 
	 * @param kdeBuilder
	 *            the kde builder
	 * @return the distribution configurable builder used for method chaining
	 */
	public DistributionConfigurableBuilder setKDEBuilder(
			KDEConfigurableBuilder kdeBuilder) {
		this.kdeBuilder = kdeBuilder;
		return this;
	}

	/**
	 * set the builder for kde distributions using a configuration containing
	 * the desired settings
	 * 
	 * @param kdeConfiguration
	 *            the kde configuration
	 * @return the distribution configurable builder used for method chaining
	 */
	public DistributionConfigurableBuilder setKDEBuilder(
			Configuration<KDEConfigurableBuilder> kdeConfiguration) {
		return setKDEBuilder(new KDEConfigurableBuilder(kdeConfiguration));
	}

	/**
	 * Gets the kDE configurable builder.
	 * 
	 * @return the kDE configurable builder
	 */
	public KDEConfigurableBuilder getKDEConfigurableBuilder() {
		return kdeBuilder;
	}

	/**
	 * Gets the bins.
	 * 
	 * @return the bins
	 */
	public int getBins() {
		return this.bins;
	}

	/**
	 * Gets the dist type.
	 * 
	 * @return the dist type
	 */
	public DistributionType getDistType() {
		return distType;
	}

	/**
	 * Gets the kde builder.
	 * 
	 * @return the kde builder
	 */
	public KDEConfigurableBuilder getKdeBuilder() {
		return kdeBuilder;
	}

	/**
	 * Option info, prints command line params representing the current
	 * configuration
	 * 
	 * @return the string
	 */
	public static String optionInfo() {
		StringBuilder builder = new StringBuilder();
		for (Option option : options) {
			builder.append("-" + option.getShortName() + ": "
					+ option.getDescription() + ", ");
		}
		builder.replace(builder.length() - 2, builder.length(), "");
		return builder.toString();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.util.option.Configurable#getConfiguration()
	 */
	@Override
	public Configuration<DistributionConfigurableBuilder> getConfiguration() {
		Configuration<DistributionConfigurableBuilder> conf = new Configuration<DistributionConfigurableBuilder>(
				options);
		populateConfiguration(conf);
		return conf;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.option.Configurable#populateConfiguration(com.parallax
	 * .ml.util.option.Configuration)
	 */
	@Override
	public Configuration<DistributionConfigurableBuilder> populateConfiguration(
			Configuration<DistributionConfigurableBuilder> conf) {
		conf.addEnumValueOnShortName("D", distType);
		conf.addFloatValueOnShortName("a", alpha);
		conf.addIntegerValueOnShortName("b", bins);
		conf.addConfigurableValueOnShortName("K", kdeBuilder.getConfiguration());
		return conf;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.util.option.Configurable#configure(com.parallax.ml.util
	 * .option.Configuration)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void configure(
			Configuration<DistributionConfigurableBuilder> configuration) {
		setDistributionType((DistributionType) configuration
				.enumFromShortName("D"));
		setAlpha(configuration.floatOptionFromShortName("a"));
		setBins(configuration.integerOptionFromShortName("b"));
		setKDEBuilder((Configuration<KDEConfigurableBuilder>) configuration
				.configurationFromShortName("K"));
	}

	/**
	 * The Class DistributionConfigurableOptions; settings that can be tuned to
	 * configure {@line DistributionConfigurableBuilder} options are:
	 * 
	 * D/distribution : the type of distribution to be considered, an enum from
	 * {@link DistributionType}<br>
	 * a/alpha : (>0) : alpha used for laplace smoothing<br>
	 * b/bins : (integer, > 0), # bins used for histogram distributions<br>
	 * K/kdeconfiguration: configuration representing the desired settings of a
	 * kde distribution builder {@link KDEConfigurableBuilder}.
	 * 
	 */
	public static class DistributionConfigurableOptions extends
			OptionSet<DistributionConfigurableBuilder> {
		{
			addOption(new EnumOption<DistributionType>("D", "distribution",
					true, "the type of distribution to be used. options are: "
							+ Arrays.toString(DistributionType.values()),
					DistributionType.class, DistributionType.MULTINOMIAL));
			addOption(new FloatOption("a", "alpha",
					"alpha used for laplace smoothing", 1., true,
					new GreaterThanOrEqualsValueBound(0),
					new LessThanOrEqualsValueBound(10000)));
			addOption(new IntegerOption("b", "bins",
					"bins used in histogram distributions", 100, false,
					new GreaterThanOrEqualsValueBound(2),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new ConfigurableOption<KDEConfigurableBuilder>("K",
					"kdeconfiguration", true,
					"configurable builder for kdd distributions. options:"
							+ KDEConfigurableBuilder.optionInfo(),
					new Configuration<KDEConfigurableBuilder>(
							KDEConfigurableBuilder.options)));
		}
	}

}
