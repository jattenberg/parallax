/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import com.dsi.parallax.ml.util.bounds.Bounds;
import com.google.common.collect.Iterators;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * OptionSet creates 3 collection: options,shortNameToOptions and
 * longNameToOptions options stores all of option objects shortNameToOptions as
 * Map sets short name as key and option objects as value longNameToOptions as
 * Map sets long name as key and option objects as value at the same time, user
 * can get all of option objects.
 * 
 * @author Josh Attenberg
 */
public class OptionSet<C extends Configurable<C>> implements Iterable<Option> {
	protected Set<Option> options;
	protected Map<String, Option> shortNameToOptions;
	protected Map<String, Option> longNameToOptions;

	/**
	 * Class constructor.
	 */
	public OptionSet() {
		this.options = Sets.newHashSet();
		shortNameToOptions = Maps.newHashMap();
		longNameToOptions = Maps.newHashMap();
	}

	/**
	 * Class constructor specifying multiple Option to create
	 * 
	 * @param options
	 *            multiple options
	 */
	public OptionSet(Option... options) {
		this();
		for (Option o : options)
			addOption(o);
	}

	/**
	 * Class constructor specifying Option collection to create
	 * 
	 * @param options
	 *            option collection
	 */
	public OptionSet(Collection<Option> options) {
		this();
		for (Option o : options)
			addOption(o);
	}

	/**
	 * The method adds new Option to options, shortNameToOptions and
	 * longNameToOptions overwrites any option withe same short or long name
	 * 
	 * @param o
	 *            Option
	 */
	public void addOption(Option o) {
		if (shortNameToOptions.containsKey(o.shortName)) {
			options.remove(shortNameToOptions.get(o.shortName));
			shortNameToOptions.remove(o.shortName);
		}

		if (longNameToOptions.containsKey(o.longName)) {
			options.remove(longNameToOptions.get(o.longName));
			longNameToOptions.remove(o.longName);
		}

		options.add(o);
		shortNameToOptions.put(o.getShortName(), o);
		longNameToOptions.put(o.getLongName(), o);
	}

	/**
	 * get the lower and upper bounds of valid values for an integer option
	 * 
	 * @param key
	 *            short name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getIntegerBoundsFromShortName(String key) {
		checkArgument(shortNameToOptions.containsKey(key),
				"%s is not a valid short key name", key);
		checkArgument(shortNameToOptions.get(key) instanceof IntegerOption,
				"option %s is not an integer option. (actually a %s", key,
				shortNameToOptions.get(key).getClass().getName());

		IntegerOption opt = (IntegerOption) shortNameToOptions.get(key);
		return opt.getBounds();
	}

	/**
	 * get the lower and upper bounds of valid values for an integer option
	 * 
	 * @param key
	 *            long name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getIntegerBoundsFromLongName(String key) {
		checkArgument(longNameToOptions.containsKey(key),
				"%s is not a valid long key name", key);
		checkArgument(longNameToOptions.get(key) instanceof IntegerOption,
				"option %s is not an integer option. (actually a %s", key,
				longNameToOptions.get(key).getClass().getName());

		IntegerOption opt = (IntegerOption) longNameToOptions.get(key);
		return opt.getBounds();
	}

	/**
	 * get the lower and upper bounds of valid values for an float option
	 * 
	 * @param key
	 *            short name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getFloatBoundsFromShortName(String key) {
		checkArgument(shortNameToOptions.containsKey(key),
				"%s is not a valid short key name", key);
		checkArgument(shortNameToOptions.get(key) instanceof FloatOption,
				"option %s is not an float option. (actually a %s", key,
				shortNameToOptions.get(key).getClass().getName());

		FloatOption opt = (FloatOption) shortNameToOptions.get(key);
		return opt.getBounds();
	}

	/**
	 * get the lower and upper bounds of valid values for an float option
	 * 
	 * @param key
	 *            long name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getFloatBoundsFromLongName(String key) {
		checkArgument(longNameToOptions.containsKey(key),
				"%s is not a valid long key name", key);
		checkArgument(longNameToOptions.get(key) instanceof FloatOption,
				"option %s is not an float option. (actually a %s", key,
				longNameToOptions.get(key).getClass().getName());

		FloatOption opt = (FloatOption) longNameToOptions.get(key);
		return opt.getBounds();
	}

	/**
	 * The method gets Options collection
	 * 
	 * @return multiple Option
	 */
	public Collection<Option> getOptions() {
		return options;
	}

	/**
	 * The method encapsulates options collection to iterator
	 * 
	 * @return multiple Option
	 */
	@Override
	public Iterator<Option> iterator() {
		return Iterators.unmodifiableIterator(options.iterator());
	}

	/**
	 * get a copy of an option set
	 * 
	 * @return
	 */
	OptionSet<C> copyOptionSet() {
		OptionSet<C> os = new OptionSet<C>();
		for (Option o : options)
			os.addOption(o.copyOption());
		return os;
	}

	/**
	 * get the available list of short option names
	 * 
	 * @return
	 */
	public Set<String> getShortNames() {
		return shortNameToOptions.keySet();
	}
}
