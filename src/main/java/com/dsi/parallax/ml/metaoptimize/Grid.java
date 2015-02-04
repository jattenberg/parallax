/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize;

import com.dsi.parallax.ml.util.option.Configurable;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.Option;
import com.google.common.collect.Lists;

import java.util.Iterator;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/**
 * a class for representing the points to be searched over.
 *
 * @param <C> the generic type
 * @author jattenberg
 */
public abstract class Grid<C extends Configurable<C>> implements Iterator<Configuration<C>> {

	/** The config. */
	protected final Configuration<C> config;
	
	/** The options to optimize. */
	protected final List<Option> optionsToOptimize;

	/**
	 * Instantiates a new grid.
	 *
	 * @param config the config
	 * @param optionShortNames the option short names
	 */
	protected Grid(Configuration<C> config, String[] optionShortNames) {
		List<Option> tOpts = config.getOptions();
		List<Option> selected = Lists.newArrayList();

		for (String shortName : optionShortNames) {
			boolean contains = false;
			for (Option opt : tOpts) {
				if (opt.getShortName().equals(shortName)) {
					selected.add(opt);
					contains = true;
				}
			}
			checkArgument(
					contains,
					"%s is an invalid configuration short name for this problem",
					shortName);
		}
		this.config = config;
		this.optionsToOptimize = selected;
	}

	/**
	 * Instantiates a new grid.
	 *
	 * @param config the config
	 * @param optionsToOptimize the options to optimize
	 */
	protected Grid(Configuration<C> config, List<Option> optionsToOptimize) {
		this.config = config;
		this.optionsToOptimize = optionsToOptimize;
	}

	/**
	 * Instantiates a new grid.
	 *
	 * @param config the config
	 */
	protected Grid(Configuration<C> config) {
		this(config, config.getOptimizableOptions());
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#remove()
	 */
	@Override
	public void remove() {
		throw new UnsupportedOperationException(
				"remove isnt a supported operation for type grid");

	}
}
