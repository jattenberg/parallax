/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.mercerkernels;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Arrays;

import com.dsi.parallax.ml.util.bounds.GreaterThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.option.AbstractConfigurable;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.EnumOption;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.Option;
import com.dsi.parallax.ml.util.option.OptionSet;

// TODO: Auto-generated Javadoc
/**
 * The Class KernelConfigurableBuilder.
 */
public class KernelConfigurableBuilder extends
		AbstractConfigurable<KernelConfigurableBuilder> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 579459681606259659L;
	
	/** The kernel type. */
	private KernelType kernelType = KernelType.GRBF;
	
	/** The sigma. */
	private double sigma = 2.;
	
	/** The gamma. */
	private double gamma = 2.;
	
	/** The degree. */
	private double degree = 2;

	/** The options. */
	public static OptionSet<KernelConfigurableBuilder> options = new KernelConfigurableOptions();

	/**
	 * Instantiates a new kernel configurable builder.
	 */
	public KernelConfigurableBuilder() {
		this(new Configuration<KernelConfigurableBuilder>(options));
	}

	/**
	 * Instantiates a new kernel configurable builder.
	 *
	 * @param config the config
	 */
	public KernelConfigurableBuilder(
			Configuration<KernelConfigurableBuilder> config) {
		configure(config);
	}

	/**
	 * Sets the sigma.
	 *
	 * @param sigma the sigma
	 * @return the kernel configurable builder
	 */
	public KernelConfigurableBuilder setSigma(double sigma) {
		checkArgument(sigma > 0, "sigma must be > 0 input: %s", sigma);
		this.sigma = sigma;
		return this;
	}

	/**
	 * Sets the gamma.
	 *
	 * @param gamma the gamma
	 * @return the kernel configurable builder
	 */
	public KernelConfigurableBuilder setGamma(double gamma) {
		checkArgument(gamma > 0, "gamma must be > 0 input: %s", gamma);
		this.gamma = gamma;
		return this;
	}

	/**
	 * Sets the degree.
	 *
	 * @param degree the degree
	 * @return the kernel configurable builder
	 */
	public KernelConfigurableBuilder setDegree(double degree) {
		checkArgument(degree > 0, "degree must be > 0 input: %s", degree);
		this.degree = degree;
		return this;
	}

	/**
	 * Sets the kernel type.
	 *
	 * @param kernelType the kernel type
	 * @return the kernel configurable builder
	 */
	public KernelConfigurableBuilder setKernelType(KernelType kernelType) {
		this.kernelType = kernelType;
		return this;
	}

	/**
	 * Configuration from kernel.
	 *
	 * @param kernel the kernel
	 * @return the configuration
	 */
	public static Configuration<KernelConfigurableBuilder> configurationFromKernel(
			Kernel kernel) {
		Configuration<KernelConfigurableBuilder> config = new Configuration<KernelConfigurableBuilder>(
				options);
		if (kernel.getClass().isAssignableFrom(GaussianRBFKernel.class)) {
			config.addEnumValueOnShortName("K", KernelType.GRBF);
			GaussianRBFKernel kern = (GaussianRBFKernel) kernel;
			config.addFloatValueOnShortName("s", kern.getSigma());
		} else if (kernel.getClass().isAssignableFrom(LinearKernel.class)) {
			config.addEnumValueOnShortName("K", KernelType.LINEAR);
		} else if (kernel.getClass().isAssignableFrom(PolynomialKernel.class)) {
			config.addEnumValueOnShortName("K", KernelType.POLYNOMIAL);
			config.addFloatValueOnShortName("D",
					((PolynomialKernel) kernel).getDegree());
		} else if (kernel.getClass().isAssignableFrom(
				InhomogeneousPolynomialKernel.class)) {
			config.addEnumValueOnShortName("K", KernelType.IPOLY);
			config.addFloatValueOnShortName("D",
					((InhomogeneousPolynomialKernel) kernel).getDegree());
		} else if (kernel.getClass().isAssignableFrom(RBFKernel.class)) {
			config.addEnumValueOnShortName("K", KernelType.RBF);
			config.addFloatValueOnShortName("g",
					((RBFKernel) kernel).getGamma());
		}

		return config;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.util.option.Configurable#getConfiguration()
	 */
	@Override
	public Configuration<KernelConfigurableBuilder> getConfiguration() {
		Configuration<KernelConfigurableBuilder> conf = new Configuration<KernelConfigurableBuilder>(
				options);
		populateConfiguration(conf);
		return conf;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.util.option.Configurable#populateConfiguration(com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public Configuration<KernelConfigurableBuilder> populateConfiguration(
			Configuration<KernelConfigurableBuilder> conf) {
		conf.addFloatValueOnShortName("D", degree);
		conf.addFloatValueOnShortName("s", sigma);
		conf.addFloatValueOnShortName("g", gamma);
		conf.addEnumValueOnShortName("K", kernelType);
		return conf;
	}

	/* (non-Javadoc)
	 * @see com.parallax.ml.util.option.Configurable#configure(com.parallax.ml.util.option.Configuration)
	 */
	@Override
	public void configure(Configuration<KernelConfigurableBuilder> configuration) {
		setKernelType((KernelType) configuration.enumFromShortName("K"));
		setSigma(configuration.floatOptionFromShortName("s"));
		setGamma(configuration.floatOptionFromShortName("g"));
		setDegree(configuration.floatOptionFromShortName("D"));
	}

	/**
	 * Builds the kernel.
	 *
	 * @return the kernel
	 */
	public Kernel buildKernel() {
		switch (kernelType) {
		case LINEAR:
			return new LinearKernel();
		case POLYNOMIAL:
			return new PolynomialKernel(degree);
		case IPOLY:
			return new InhomogeneousPolynomialKernel(degree);
		case RBF:
			return new RBFKernel(gamma);
		case GRBF:
			return new GaussianRBFKernel(sigma);
		default:
			throw new IllegalArgumentException(kernelType
					+ " is not a valid kernel type, options:"
					+ Arrays.toString(KernelType.values()));
		}
	}

	/**
	 * Gets the default configuration.
	 *
	 * @return the default configuration
	 */
	public static Configuration<KernelConfigurableBuilder> getDefaultConfiguration() {
		return new Configuration<KernelConfigurableBuilder>(options);
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

	/**
	 * The Class KernelConfigurableOptions.
	 */
	public static class KernelConfigurableOptions extends
			OptionSet<KernelConfigurableBuilder> {
		{
			addOption(new EnumOption<KernelType>("K", "kernel", true,
					"type of kernel to be used. options:"
							+ Arrays.toString(KernelType.values()),
					KernelType.class, KernelType.POLYNOMIAL));
			addOption(new FloatOption("s", "sigma",
					"sigma parameter for gaussian rbf kernel", 0.2, true,
					new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("g", "gamma",
					"gamma parameter for rbf kernel", 0.1, true,
					new GreaterThanValueBound(0),
					new LessThanOrEqualsValueBound(BIGVAL)));
			addOption(new FloatOption("D", "degree",
					"degree for polynomial kernels", 2, true,
					new GreaterThanOrEqualsValueBound(1),
					new LessThanOrEqualsValueBound(1000)));
		}
	}

}
