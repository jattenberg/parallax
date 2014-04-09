/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import static com.google.common.base.Preconditions.checkArgument;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.dsi.parallax.ml.util.bounds.Bounds;
import com.google.common.base.Joiner;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

/**
 * registers to op
 * 
 * @author jattenberg
 */
public class Configuration<C extends Configurable<C>> {

	protected Map<String, Enum<?>> enumOptions;
	protected Map<String, Double> floatOptions;
	protected Map<String, Integer> integerOptions;
	protected Map<String, Boolean> booleanOptions;
	protected Map<String, Configuration<?>> configurableOptions;
	protected Map<String, NestedConfiguration<?, ?, ?>> nestedConfigurableOptions;

	private BiMap<String, String> shortNameToLongName;
	private final OptionSet<C> options;
	protected Options cliOpts;
	private final static Joiner joiner = Joiner.on(" ").skipNulls();

	/**
	 * Class constructor
	 */
	protected Configuration() {
		booleanOptions = Maps.newHashMap();
		floatOptions = Maps.newHashMap();
		enumOptions = Maps.newHashMap();
		integerOptions = Maps.newHashMap();
		configurableOptions = Maps.newHashMap();
		nestedConfigurableOptions = Maps.newHashMap();

		shortNameToLongName = HashBiMap.create();
		options = new OptionSet<C>();
		cliOpts = new Options();
	}

	/**
	 * Class constructor specifying OptionSet to create
	 * 
	 * @param options
	 *            OptionSet
	 */
	public Configuration(OptionSet<C> options) {
		this();
		for (Option opt : options) {
			addOption(opt);
		}
	}

	/**
	 * The method adds new Option to OptionSet
	 * 
	 * @param o
	 *            Option
	 */
	public void addOption(Option o) {

		// check for duplicates
		checkArgument(!shortNameToLongName.containsKey(o.shortName),
				"%s is already a used short option key", o.shortName);
		checkArgument(!shortNameToLongName.inverse().containsKey(o.longName),
				"%s is already a used long option key", o.longName);

		insertOption(o);
	}

	/**
	 * inserts an option into the appropriate data structures for later use.
	 * 
	 * @param o
	 *            Option to be inserted
	 */
	private void insertOption(Option o) {
		options.addOption(o);
		shortNameToLongName.put(o.shortName, o.longName);
		cliOpts.addOption(o.getShortName(), o.getLongName(),
				o.getType() == OptionType.BOOLEAN ? false : true,
				o.getDescription());

		switch (o.getType()) {
		case FLOAT:
			FloatOption fo = (FloatOption) o;
			floatOptions.put(fo.longName, fo.getDEFAULT());
			break;
		case INTEGER:
			IntegerOption io = (IntegerOption) o;
			integerOptions.put(io.longName, io.getDEFAULT());
			break;
		case ENUM:
			EnumOption<?> eo = (EnumOption<?>) o;
			enumOptions.put(eo.longName, eo.getDEFAULTE());
			break;
		case BOOLEAN:
			BooleanOption bo = (BooleanOption) o;
			booleanOptions.put(bo.longName, bo.getDEFAULTB());
			break;
		case CONFIGURABLE:
			ConfigurableOption<?> co = (ConfigurableOption<?>) o;
			configurableOptions.put(co.getLongName(), co.getDefaultConfig());
			break;
		case NESTEDCONFIGURABLE:
			NestedConfigurableOption<?, ?, ?> nco = (NestedConfigurableOption<?, ?, ?>) o;
			nestedConfigurableOptions.put(nco.getLongName(),
					(NestedConfiguration<?, ?, ?>) nco.getDefaultConfig());
		}
	}

	/**
	 * populates the configuration from arguments TODO: better handling of enum
	 * type
	 * 
	 * @param args
	 *            arguments
	 * @throws ParseException
	 *             parse exception
	 * @throws IllegalArgumentException
	 *             illegal argument exception
	 * @throws IllegalAccessException
	 *             illegal access exception
	 * @throws InvocationTargetException
	 *             invocation target exception
	 * @throws SecurityException
	 *             security exception
	 * @throws NoSuchMethodException
	 *             no such method exception
	 */
	@SuppressWarnings("unchecked")
	public void optionValuesFromArgs(String[] args) throws ParseException,
			IllegalArgumentException, IllegalAccessException,
			InvocationTargetException, SecurityException, NoSuchMethodException {
		CommandLineParser parser = new GnuParser();
		CommandLine line = parser.parse(cliOpts, args);
		for (Option o : options) {
			if (line.hasOption(o.getShortName())) {
				switch (o.getType()) {
				case FLOAT:
					FloatOption fo = (FloatOption) o;
					double val = Double.parseDouble(line.getOptionValue(o
							.getShortName()));
					floatOptions.put(fo.longName, fo.checkCurrent(val));
					break;
				case INTEGER:
					IntegerOption io = (IntegerOption) o;
					int ival = Integer.parseInt(line.getOptionValue(o
							.getShortName()));
					integerOptions.put(io.longName, io.checkCurrent(ival));
					break;
				case ENUM:
					EnumOption<?> eo = (EnumOption<?>) o;
					Method m = eo.getDEFAULTE().getClass()
							.getMethod("valueOf", String.class);

					@SuppressWarnings("rawtypes")
					Enum eval = (Enum<?>) m.invoke(eo.getDEFAULTE().getClass(),
							line.getOptionValue(o.getShortName()));
					enumOptions.put(eo.longName, eo.checkCurrent(eval));
					break;
				case BOOLEAN:
					BooleanOption bo = (BooleanOption) o;
					boolean bval = bo.getDEFAULTB() ? false : true;
					booleanOptions.put(bo.longName, bval);
					break;
				case CONFIGURABLE:
					ConfigurableOption<?> co = (ConfigurableOption<?>) o;
					Configuration<?> config = co.getDefaultConfig();
					config.optionValuesFromArgs(line.getOptionValues(o
							.getShortName()));
					configurableOptions.put(co.longName, config);
					break;
				case NESTEDCONFIGURABLE:
					NestedConfigurableOption<?, ?, ?> no = (NestedConfigurableOption<?, ?, ?>) o;
					NestedConfiguration<?, ?, ?> nestedConfig = (NestedConfiguration<?, ?, ?>) no
							.getDefaultConfig();
					nestedConfig.optionValuesFromArgs(line.getOptionValues(o
							.getShortName()));
					nestedConfigurableOptions.put(no.longName, nestedConfig);
					break;
				}
			}
		}
	}

	/**
	 * The method gets all options from Options collection
	 * 
	 * @return Options
	 */
	public Options getCliOpts() {
		return cliOpts;
	}

	/**
	 * The method gets all arguments from Options
	 * 
	 * @return arguments
	 */
	public String[] getArgumentsFromOpts() {
		List<String> parts = new ArrayList<String>();
		for (Option o : options) {
			switch (o.getType()) {
			case BOOLEAN:
				BooleanOption bo = (BooleanOption) o;
				if (booleanOptions.get(bo.getLongName()) != bo.getDEFAULTB())
					parts.add("-" + o.getShortName());
				break;
			case FLOAT:
				FloatOption fo = (FloatOption) o;
				parts.add("-" + fo.getShortName());
				parts.add("" + floatOptions.get(fo.getLongName()));
				break;
			case INTEGER:
				IntegerOption io = (IntegerOption) o;
				parts.add("-" + io.getShortName());
				parts.add("" + integerOptions.get(io.getLongName()));
				break;
			case ENUM:
				EnumOption<?> eo = (EnumOption<?>) o;
				parts.add("-" + eo.getShortName());
				parts.add(enumOptions.get(eo.getLongName()).toString());
				break;
			case CONFIGURABLE:
			case NESTEDCONFIGURABLE:
				ConfigurableOption<?> co = (ConfigurableOption<?>) o;
				parts.add("-" + co.getShortName());
				parts.add("\""
						+ joiner.join(configurableOptions.get(co.getLongName())
								.getArgumentsFromOpts()) + "\"");
				break;

			}
		}
		return convertObjectArray2StringArray(parts.toArray());
	}

	private String[] convertObjectArray2StringArray(Object[] objArray) {
		String[] stringArray = new String[objArray.length];
		for (int i = 0; i < objArray.length; i++) {
			stringArray[i] = objArray[i].toString();
		}
		return stringArray;
	}

	/**
	 * get a set containing all of the options available in the configuration
	 * 
	 * @return
	 */
	public List<Option> getOptions() {
		List<Option> outOptions = Lists.newArrayList();
		for (Option option : options)
			outOptions.add(option);
		return outOptions;
	}

	public List<Option> getOptimizableOptions() {
		List<Option> outOptions = Lists.newArrayList();
		for (Option option : options)
			if (option.isOptimizable)
				outOptions.add(option);
		return outOptions;
	}

	/**
	 * The method adds Float option's value to shortNameToLongName BiMap on
	 * short name
	 * 
	 * @param s
	 *            short name
	 * @param v
	 *            configuration of associated option
	 * @return Configuration
	 */
	public <X extends Enum<X> & NestedConfigurableType<Y, X>, Y extends NestedConfigurable<Y, X>, Z extends ParentNestedConfigurable<X, Y, Z>> Configuration<C> addNestedConfigurableValueOnShortName(
			String s, NestedConfiguration<X, Y, Z> c) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		addNestedConfigurableValueOnLongName(shortNameToLongName.get(s), c);
		return this;
	}

	/**
	 * The method add Float option's value to shortNameToLongName BiMap on long
	 * name
	 * 
	 * @param s
	 *            long name
	 * @param v
	 *            configuration of associated option
	 * @return Configuration
	 */
	public <X extends Enum<X> & NestedConfigurableType<Y, X>, Y extends NestedConfigurable<Y, X>, Z extends ParentNestedConfigurable<X, Y, Z>> Configuration<C> addNestedConfigurableValueOnLongName(
			String s, NestedConfiguration<X, Y, Z> c) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined shortname: %s", s);
		nestedConfigurableOptions.put(s, c);
		return this;
	}

	/**
	 * The method adds Float option's value to shortNameToLongName BiMap on
	 * short name
	 * 
	 * @param s
	 *            short name
	 * @param v
	 *            configuration of associated option
	 * @return Configuration
	 */
	public <T extends Configurable<T>> Configuration<C> addConfigurableValueOnShortName(
			String s, Configuration<T> c) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		addConfigurableValueOnLongName(shortNameToLongName.get(s), c);
		return this;
	}

	/**
	 * The method add Float option's value to shortNameToLongName BiMap on long
	 * name
	 * 
	 * @param s
	 *            long name
	 * @param v
	 *            configuration of associated option
	 * @return Configuration
	 */
	public <T extends Configurable<T>> Configuration<C> addConfigurableValueOnLongName(
			String s, Configuration<T> c) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined shortname: %s", s);
		configurableOptions.put(s, c);
		return this;
	}

	/**
	 * The method adds Float option's value to shortNameToLongName BiMap on
	 * short name
	 * 
	 * @param s
	 *            short name
	 * @param v
	 *            float value
	 * @return Configuration
	 */
	public Configuration<C> addFloatValueOnShortName(String s, double v) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		addFloatValueOnLongName(shortNameToLongName.get(s), v);
		return this;
	}

	/**
	 * The method add Float option's value to shortNameToLongName BiMap on long
	 * name
	 * 
	 * @param s
	 *            long name
	 * @param v
	 *            float option value
	 * @return Configuration
	 */
	public Configuration<C> addFloatValueOnLongName(String s, double v) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined shortname: %s", s);
		floatOptions.put(s, v);
		return this;
	}

	/**
	 * The method adds Integer option's value to shortNameToLongName on short
	 * name
	 * 
	 * @param s
	 *            short name
	 * @param v
	 *            Integer option's value
	 * @return Configuration
	 */
	public Configuration<C> addIntegerValueOnShortName(String s, int v) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		addIntegerValueOnLongName(shortNameToLongName.get(s), v);
		return this;
	}

	/**
	 * The method adds Integer option's value to shortNameToLongName on long
	 * name
	 * 
	 * @param s
	 *            long name
	 * @param v
	 *            Integer option's value
	 * @return Configuration
	 */
	public Configuration<C> addIntegerValueOnLongName(String s, int v) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined shortname: %s", s);
		integerOptions.put(s, v);
		return this;
	}

	/**
	 * The method adds Boolean option's value to shortNameToLongName on short
	 * name
	 * 
	 * @param s
	 *            short name
	 * @param v
	 *            Boolean option's value
	 * @return Configuration
	 */
	public Configuration<C> addBooleanValueOnShortName(String s, boolean v) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		addBooleanValueOnLongName(shortNameToLongName.get(s), v);
		return this;
	}

	/**
	 * The method adds Boolean option's value to shortNameToLongName on long
	 * name
	 * 
	 * @param s
	 *            long name
	 * @param v
	 *            Boolean option's value
	 * @return Configuration
	 */
	public Configuration<C> addBooleanValueOnLongName(String s, boolean v) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined shortname: %s", s);
		booleanOptions.put(s, v);
		return this;
	}

	/**
	 * The method adds Enum option's value to shortNameToLongName on short name
	 * 
	 * @param s
	 *            short name
	 * @param v
	 *            Enum option's value
	 * @return Configuration
	 */
	public Configuration<C> addEnumValueOnShortName(String s, Enum<?> v) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		addEnumValueOnLongName(shortNameToLongName.get(s), v);
		return this;
	}

	/**
	 * The method adds Enum option's value to shortNameToLongName on long name
	 * 
	 * @param s
	 *            long name
	 * @param v
	 *            Enum option's value
	 * @return Configuration
	 */
	public Configuration<C> addEnumValueOnLongName(String s, Enum<?> v) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined shortname: %s", s);
		enumOptions.put(s, v);
		return this;
	}

	public NestedConfiguration<?, ?, ?> nestedConfigurationFromShortName(
			String s) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		return nestedConfigurationFromLongName(shortNameToLongName.get(s));
	}

	public NestedConfiguration<?, ?, ?> nestedConfigurationFromLongName(String s) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined longname: %s", s);
		return nestedConfigurableOptions.get(s);
	}

	/**
	 * The method gets the value of a configuration from it's short name
	 * 
	 * @param s
	 *            short name
	 * @return Configuration option
	 */
	public Configuration<?> configurationFromShortName(String s) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		return configurationFromLongName(shortNameToLongName.get(s));
	}

	/**
	 * The method gets the value of a configuration from it's long name
	 * 
	 * @param s
	 *            long name
	 * @return Configuration of option
	 */
	public Configuration<?> configurationFromLongName(String s) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined longname: %s", s);
		return configurableOptions.get(s);
	}

	/**
	 * The method gets Enum option by short name
	 * 
	 * @param s
	 *            short name
	 * @return Enum option
	 */
	public Enum<?> enumFromShortName(String s) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		return enumFromLongName(shortNameToLongName.get(s));
	}

	/**
	 * The method gets Enum option by long name
	 * 
	 * @param s
	 *            long name
	 * @return Enum option
	 */
	public Enum<?> enumFromLongName(String s) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined longname: %s", s);
		return enumOptions.get(s);
	}

	/**
	 * The method gets Integer Option by short name
	 * 
	 * @param s
	 *            short name
	 * @return Integer option
	 */
	public int integerOptionFromShortName(String s) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		return integerFromLongName(shortNameToLongName.get(s));
	}

	/**
	 * The method gets integer option by long name
	 * 
	 * @param s
	 *            long name
	 * @return Integer option
	 */
	public int integerFromLongName(String s) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined longname: %s", s);
		return integerOptions.get(s);
	}

	/**
	 * The method gets Float Option by short name
	 * 
	 * @param s
	 *            short name
	 * @return Float Option
	 */
	public double floatOptionFromShortName(String s) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		return floatFromLongName(shortNameToLongName.get(s));
	}

	/**
	 * The method gets Float Option by long name
	 * 
	 * @param s
	 *            long name
	 * @return Float Option
	 */
	public double floatFromLongName(String s) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined longname: %s", s);
		return floatOptions.get(s);
	}

	/**
	 * The method gets Boolean Option by short name
	 * 
	 * @param s
	 *            short name
	 * @return Boolean Option
	 */
	public boolean booleanOptionFromShortName(String s) {
		checkArgument(shortNameToLongName.containsKey(s),
				"undefined shortname: %s", s);
		return booleanFromLongName(shortNameToLongName.get(s));
	}

	/**
	 * The method gets Boolean Option by long name
	 * 
	 * @param s
	 *            long name
	 * @return Boolean Option
	 */
	public boolean booleanFromLongName(String s) {
		checkArgument(shortNameToLongName.inverse().containsKey(s),
				"undefined longname: %s", s);
		return booleanOptions.get(s);
	}

	/**
	 * The method checks if shortNameToLongName contains short name
	 * 
	 * @param s
	 *            short name
	 * @return boolean
	 */
	public boolean containsShortKey(String s) {
		return shortNameToLongName.containsKey(s);
	}

	/**
	 * The method checks if shortNameToLongName contains long name
	 * 
	 * @param s
	 *            long name
	 * @return boolean
	 */
	public boolean containsLongKey(String s) {
		return shortNameToLongName.inverse().containsKey(s);
	}

	/**
	 * get the lower and upper bounds of valid values for an integer option
	 * 
	 * @param key
	 *            short name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getIntegerBoundsFromShortName(String key) {
		return options.getIntegerBoundsFromShortName(key);
	}

	/**
	 * get the lower and upper bounds of valid values for an integer option
	 * 
	 * @param key
	 *            long name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getIntegerBoundsFromLongName(String key) {
		return options.getIntegerBoundsFromLongName(key);
	}

	/**
	 * get the lower and upper bounds of valid values for an float option
	 * 
	 * @param key
	 *            short name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getFloatBoundsFromShortName(String key) {
		return options.getFloatBoundsFromShortName(key);
	}

	/**
	 * get the lower and upper bounds of valid values for an float option
	 * 
	 * @param key
	 *            long name of the option
	 * @return an array containing the (lower bound, upper bound)
	 */
	public Bounds getFloatBoundsFromLongName(String key) {
		return options.getFloatBoundsFromLongName(key);
	}

	/**
	 * The method copys all the properties of Configuration to new Configuration
	 * 
	 * @return Configuration
	 */
	public Configuration<C> copyConfiguration() {
		return new Configuration<C>(options.copyOptionSet());
	}

	public void getHelp() {
		HelpFormatter formatter = new HelpFormatter();
		formatter.printHelp(getClass().getName(), getCliOpts());
	}

	public OptionSet<C> getOptionSet() {
		return options;
	}
}
