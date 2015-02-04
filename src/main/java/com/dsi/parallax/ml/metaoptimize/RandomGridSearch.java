/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.util.RandomUtils;
import com.dsi.parallax.ml.util.option.*;

import java.util.List;

// TODO: Auto-generated Javadoc
/**
 * The Class RandomGridSearch.
 *
 * @param <C> the generic type
 */
public class RandomGridSearch<C extends Configurable<C>> extends Grid<C> {

	/**
	 * Instantiates a new random grid search.
	 *
	 * @param config the config
	 * @param optionShortNames the option short names
	 */
	public RandomGridSearch(Configuration<C> config, String[] optionShortNames) {
		super(config, optionShortNames);
	}

	/**
	 * Instantiates a new random grid search.
	 *
	 * @param config the config
	 * @param optionsToOptimize the options to optimize
	 */
	public RandomGridSearch(Configuration<C> config, List<Option> optionsToOptimize) {
		super(config, optionsToOptimize);
	}

	/**
	 * Instantiates a new random grid search.
	 *
	 * @param config the config
	 */
	public RandomGridSearch(Configuration<C> config) {
		super(config);
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#hasNext()
	 */
	@Override
	public boolean hasNext() {
		return true;
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#next()
	 */
	@Override
	public Configuration<C> next() {
		Configuration<C> nextConfig = config.copyConfiguration();

		for (Option o : optionsToOptimize) {
			switch (o.getType()) {
			case FLOAT:
				FloatOption fo = (FloatOption) o;
				double[] fBounds = fo.getBounds().numericValues();
				double nextVal = RandomUtils.INSTANCE.nextUniform(fBounds[0], fBounds[1]);
				nextConfig.addFloatValueOnShortName(fo.getShortName(),
						nextVal);
				break;
			case INTEGER:
				IntegerOption io = (IntegerOption) o;
				int[] iBounds = io.getBounds().integerValues();
				nextConfig.addIntegerValueOnShortName(
						io.getShortName(),
						iBounds[0]
								+ MLUtils.GENERATOR.nextInt(iBounds[1]
										- iBounds[0] + 1));
				break;
			case ENUM:
				EnumOption<?> eo = (EnumOption<?>) o;
				List<?> values = eo.getValueList();
				int index = RandomUtils.INSTANCE.nextInt(values.size());
				nextConfig.addEnumValueOnShortName(eo.getShortName(),
						(Enum<?>)values.get(index));
				break;
			case BOOLEAN:
				BooleanOption bo = (BooleanOption) o;
				nextConfig.addBooleanValueOnShortName(bo.getShortName(),
						MLUtils.GENERATOR.nextBoolean());
				break;
			}
		}

		return nextConfig;
	}

}
