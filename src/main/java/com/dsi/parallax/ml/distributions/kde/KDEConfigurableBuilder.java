package com.dsi.parallax.ml.distributions.kde;

import java.util.Arrays;

import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.AbstractConfigurable;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.EnumOption;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.IntegerOption;
import com.dsi.parallax.ml.util.option.Option;
import com.dsi.parallax.ml.util.option.OptionSet;

// TODO: Auto-generated Javadoc
/**
 * The Class KDEConfigurableBuilder.
 */
public class KDEConfigurableBuilder extends
		AbstractConfigurable<KDEConfigurableBuilder> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1666314815608091903L;
	
	/** The kernel. */
	private KDEKernel kernel = KDEKernel.GAUSSIAN;
	
	/** The bandwidth. */
	private int bandwidth = 50;
	
	/** The distance damping. */
	private double distanceDamping = 1;

	/** The options. */
	public static OptionSet<KDEConfigurableBuilder> options = new KDEConfigurableOptions();

	/**
	 * Instantiates a new kDE configurable builder.
	 */
	public KDEConfigurableBuilder() {

	}

	/**
	 * Instantiates a new kDE configurable builder.
	 *
	 * @param conf the conf
	 */
	public KDEConfigurableBuilder(Configuration<KDEConfigurableBuilder> conf) {
		configure(conf);
	}

	/**
	 * Sets the kernel type.
	 *
	 * @param kernel the kernel
	 * @return the kDE configurable builder
	 */
	public KDEConfigurableBuilder setKernelType(KDEKernel kernel) {
		this.kernel = kernel;
		return this;
	}

	/**
	 * Sets the band width.
	 *
	 * @param bandwidth the bandwidth
	 * @return the kDE configurable builder
	 */
	public KDEConfigurableBuilder setBandWidth(int bandwidth) {
		this.bandwidth = bandwidth;
		return this;
	}

	/**
	 * Sets the distance damping.
	 *
	 * @param distanceDamping the distance damping
	 * @return the kDE configurable builder
	 */
	public KDEConfigurableBuilder setDistanceDamping(double distanceDamping) {
		this.distanceDamping = distanceDamping;
		return this;
	}

	/**
	 * Gets the kernel.
	 *
	 * @return the kernel
	 */
	public KDEKernel getKernel() {
		return kernel;
	}

	/**
	 * Gets the bandwidth.
	 *
	 * @return the bandwidth
	 */
	public int getBandwidth() {
		return bandwidth;
	}

	/**
	 * Gets the distance damping.
	 *
	 * @return the distance damping
	 */
	public double getDistanceDamping() {
		return distanceDamping;
	}

	/**
	 * Builds the.
	 *
	 * @param dimension the dimension
	 * @return the kDE distribution
	 */
	public KDEDistribution build(int dimension) {
		return new KDEDistribution(dimension, bandwidth, distanceDamping,
				kernel);
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.util.option.Configurable#populateConfiguration(com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public Configuration<KDEConfigurableBuilder> populateConfiguration(
			Configuration<KDEConfigurableBuilder> conf) {
		conf.addEnumValueOnShortName("K", kernel);
		conf.addIntegerValueOnShortName("b", bandwidth);
		conf.addFloatValueOnShortName("D", distanceDamping);
		return null;
	}

	/**
	 * Option info.
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

	/* (non-Javadoc)
	 * @see com.parallax.ml.util.option.Configurable#configure(com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<KDEConfigurableBuilder> configuration) {
		setKernelType((KDEKernel) configuration.enumFromShortName("K"));
		setDistanceDamping(configuration.floatOptionFromShortName("D"));
		setBandWidth(configuration.integerOptionFromShortName("b"));
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.util.option.Configurable#getConfiguration()
	 */
	@Override
	public Configuration<KDEConfigurableBuilder> getConfiguration() {
		Configuration<KDEConfigurableBuilder> conf = new Configuration<KDEConfigurableBuilder>(
				options);
		populateConfiguration(conf);
		return conf;
	}

	/**
	 * The Class KDEConfigurableOptions.
	 */
	public static class KDEConfigurableOptions extends
			OptionSet<KDEConfigurableBuilder> {
		{
			addOption(new EnumOption<KDEKernel>("K", "kernel", true,
					"Type of kernel used for kernel density estimation. options: "
							+ Arrays.toString(KDEKernel.values()),
					KDEKernel.class, KDEKernel.GAUSSIAN));
			addOption(new IntegerOption(
					"b",
					"bandwidth",
					"number of neighboring points considered in KDE estimation",
					50, true, new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("D", "damping",
					"damping factor for distances, |x - y|/damping", 1, true,
					new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
		}
	}

}
