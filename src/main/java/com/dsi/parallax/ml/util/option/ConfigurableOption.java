/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

/**
 * option that represents a class that is itself configurable
 * this makes much more sense if the option represents a class that 
 * has a constructor or a factory that takes a Configuration
 * @author jattenberg
 *
 */
public class ConfigurableOption<T extends Configurable<T>> extends Option {



	private static final long serialVersionUID = -3770056250971834427L;
	protected final Configuration<T> defaultConfig;
	
	public ConfigurableOption(String shortName, String longName,
			boolean optimizable, String desc, Configuration<T> defaultConfig) {
		super(shortName, longName, optimizable, desc);
		this.defaultConfig = defaultConfig;
	}
	
	public Configuration<T> getDefaultConfig() {
		return defaultConfig.copyConfiguration();
	}
	
	@Override
	public OptionType getType() {
		return OptionType.CONFIGURABLE;
	}

	@Override
	public Option copyOption() {
		return new ConfigurableOption<T>(shortName, longName, isOptimizable, description, defaultConfig);
	}

}
