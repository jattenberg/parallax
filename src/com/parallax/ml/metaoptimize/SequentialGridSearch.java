/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.metaoptimize;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import com.google.common.collect.Maps;
import com.parallax.ml.util.option.Configurable;
import com.parallax.ml.util.option.ConfigurableOption;
import com.parallax.ml.util.option.Configuration;
import com.parallax.ml.util.option.EnumOption;
import com.parallax.ml.util.option.FloatOption;
import com.parallax.ml.util.option.IntegerOption;
import com.parallax.ml.util.option.Option;

// TODO: Auto-generated Javadoc
/**
 * The Class SequentialGridSearch.
 *
 * @param <C> the generic type
 */
public class SequentialGridSearch<C extends Configurable<C>> extends Grid<C> {

	/** The searches per option. */
	private final int searchesPerOption;
	
	/** The epoch. */
	private int epoch;
	
	/** The max epochs. */
	private int maxEpochs;
	
	/** The configurable searcher. */
	private Map<Option, SequentialGridSearch<?>> configurableSearcher;

	/**
	 * Instantiates a new sequential grid search.
	 *
	 * @param config the config
	 * @param optionsToOptimize the options to optimize
	 * @param searchesPerOption the searches per option
	 */
	public SequentialGridSearch(Configuration<C> config,
			List<Option> optionsToOptimize, int searchesPerOption) {
		super(config, optionsToOptimize);
		checkArgument(searchesPerOption > 0,
				"must check at least 1 value per option. input: %s",
				searchesPerOption);
		this.searchesPerOption = searchesPerOption;
		initializeEpochs();
	}

	/**
	 * Instantiates a new sequential grid search.
	 *
	 * @param config the config
	 * @param optionShortNames the option short names
	 * @param searchesPerOption the searches per option
	 */
	public SequentialGridSearch(Configuration<C> config,
			String[] optionShortNames, int searchesPerOption) {
		super(config, optionShortNames);
		checkArgument(searchesPerOption > 0,
				"must check at least 1 value per option. input: %s",
				searchesPerOption);
		this.searchesPerOption = searchesPerOption;
		initializeEpochs();
	}

	/**
	 * Instantiates a new sequential grid search.
	 *
	 * @param config the config
	 * @param searchesPerOption the searches per option
	 */
	public SequentialGridSearch(Configuration<C> config, int searchesPerOption) {
		super(config);
		checkArgument(searchesPerOption > 0,
				"must check at least 1 value per option. input: %s",
				searchesPerOption);
		this.searchesPerOption = searchesPerOption;
		initializeEpochs();
	}

	/**
	 * compute the max number of epochs available.
	 */
	private void initializeEpochs() {

		maxEpochs = optionsToOptimize.size() == 0 ? 0 : 1;
		for (Option o : optionsToOptimize) {
			switch (o.getType()) {
			case INTEGER:
				IntegerOption io = (IntegerOption) o;
				int[] bounds = io.getBounds().integerValues();
				maxEpochs *= searchesPerOption > (bounds[1] - bounds[0]) ? (bounds[1] - bounds[0])
						: searchesPerOption;
				break;
			case FLOAT:
				maxEpochs *= searchesPerOption;
				break;
			case BOOLEAN:
				maxEpochs *= 2;
				break;
			case ENUM:
				EnumOption<?> eo = (EnumOption<?>) o;
				List<?> values = eo.getValueList();
				maxEpochs *= values.size();
				break;
			case CONFIGURABLE:
				ConfigurableOption<?> co = (ConfigurableOption<?>) o;
				if (null == configurableSearcher)
					configurableSearcher = Maps.newHashMap();
				@SuppressWarnings({ "unchecked", "rawtypes" })
				SequentialGridSearch configSearcher = new SequentialGridSearch(
						co.getDefaultConfig(), searchesPerOption);
				maxEpochs *= configSearcher.maxEpochs;
				configurableSearcher.put(co, configSearcher);
				break;
			}
		}
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#hasNext()
	 */
	@Override
	public boolean hasNext() {
		return epoch < maxEpochs;
	}
	
	/**
	 * used for nested configurable options-
	 * one nested optimization is one run from 0 to the
	 * nested config's max epoch count.
	 */
	private void resetEpoch() {
		epoch = 0;
	}

	/* (non-Javadoc)
	 * @see java.util.Iterator#next()
	 */
	@Override
	public Configuration<C> next() {
		if (epoch >= maxEpochs)
			throw new NoSuchElementException(
					"range of possible values (" + maxEpochs + ") exceeded");
		Configuration<C> nextConfig = config.copyConfiguration();
		int step = epoch;

		for (Option o : optionsToOptimize) {
			switch (o.getType()) {
			case FLOAT:
				FloatOption fo = (FloatOption) o;
				double[] fBounds = fo.getBounds().numericValues();
				int floatSteps = searchesPerOption;
				int floatStep = step % floatSteps;
				step = (step - floatStep) / floatSteps;
				double val = doubleStep(fBounds, floatStep, floatSteps);
				nextConfig.addFloatValueOnShortName(fo.getShortName(), val);
				break;
			case INTEGER:
				IntegerOption io = (IntegerOption) o;
				int[] bounds = io.getBounds().integerValues();
				int intSteps = searchesPerOption > (bounds[1] - bounds[0]) ? (bounds[1] - bounds[0])
						: searchesPerOption;
				int intStep = step % intSteps;
				step = (step - intStep) / intSteps;
				int intVal = intStep(bounds, intStep, intSteps);
				nextConfig
						.addIntegerValueOnShortName(io.getShortName(), intVal);
				break;
			case BOOLEAN:
				int boolStep = step % 2;
				step = (step - boolStep) / 2;
				boolean boolOpt = (boolStep == 1 ? true : false);
				nextConfig
						.addBooleanValueOnShortName(o.getShortName(), boolOpt);
				break;
			case ENUM:
				EnumOption<?> eo = (EnumOption<?>) o;
				List<?> values = eo.getValueList();
				int enumStep = step % values.size();
				step = (step - enumStep) / values.size();
				Enum<?> eValue = (Enum<?>)values.get(enumStep);
				nextConfig.addEnumValueOnShortName(eo.getShortName(), eValue);
				break;
			case CONFIGURABLE:
				ConfigurableOption<?> co = (ConfigurableOption<?>) o;
				@SuppressWarnings("rawtypes")
				SequentialGridSearch searcher = configurableSearcher.get(co);
				if(epoch > 0 && epoch % searcher.maxEpochs == 0) {
					searcher.resetEpoch();
				}
				Configuration<?> config = searcher.next();
				nextConfig.addConfigurableValueOnShortName(co.getShortName(),
						config);


				break;
			}
		}
		epoch++;
		return nextConfig;
	}

	/**
	 * assumes there are [steps] steps between the bounds, including the end
	 * points reveals the value (grid point) at step [step].
	 *
	 * @param bounds the [min, max] value of the option being considered
	 * @param step which step is currently being evaluated. 0 <= step <= steps -
	 * 1
	 * @param steps max # of steps
	 * @return the grid point
	 */
	private static double doubleStep(double[] bounds, int step, int steps) {
		checkArgument(steps > 0, "steps must be positive. given: %s", steps);
		checkArgument(step >= 0 && step < steps,
				"step should be between 0 and steps - 1, given : %s", step);

		double pct = (double) step / (double) (steps - 1);
		return (bounds[1] - bounds[0]) * pct + bounds[0];
	}

	/**
	 * assumes there are [steps] steps between the bounds, including the end
	 * points reveals the value (grid point) at step [step].
	 *
	 * @param bounds the [min, max] value of the option being considered
	 * @param step which step is currently being evaluated. 0 <= step <= steps -
	 * 1
	 * @param steps max # of steps
	 * @return the grid point
	 */
	private static int intStep(int[] bounds, int step, int steps) {
		checkArgument(
				steps > 0 && steps >= (bounds[1] - bounds[1]),
				"steps must be positive, at most the number of steps. given: %s",
				steps);
		checkArgument(step >= 0 && step < steps,
				"step should be between 0 and steps - 1, given : %s", step);

		double pct = (double) step / (double) (steps - 1);
		int val = (int) Math.round((bounds[1] - bounds[0]) * pct - bounds[0]);
		return val < bounds[0] ? bounds[0] : val > bounds[1] ? bounds[1] : val;
	}

}
