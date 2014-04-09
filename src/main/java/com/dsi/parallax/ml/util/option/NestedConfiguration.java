package com.dsi.parallax.ml.util.option;

import static com.google.common.base.Preconditions.checkArgument;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.EnumSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.ParseException;

public class NestedConfiguration<T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>>
		extends Configuration<P> {

	private T configurationOption;

	public NestedConfiguration(
			ParentNestedConfigurableOptionSet<T, C, P> options) {
		super(options);
		configurationOption = options.defaultType();
	}

	@SuppressWarnings("unchecked")
	@Override
	public void optionValuesFromArgs(String[] args) throws ParseException,
			IllegalArgumentException, IllegalAccessException,
			InvocationTargetException, SecurityException, NoSuchMethodException {

		CommandLineParser parser = new GnuParser();
		CommandLine line = parser.parse(cliOpts, args);

		if (line.hasOption(ParentNestedConfigurableOptionSet.TYPESHORT)) {
			Method m = configurationOption.getClass().getMethod("valueOf",
					String.class);

			@SuppressWarnings("rawtypes")
			Enum eval = (Enum<?>) m
					.invoke(configurationOption.getClass(),
							line.getOptionValue(ParentNestedConfigurableOptionSet.TYPESHORT));
			enumOptions.put(ParentNestedConfigurableOptionSet.TYPELONG,
					checkCurrent(eval));
			configurationOption = (T) eval;
		}

		Configuration<C> nestedConfig = configurationOption
				.getDefaultConfiguration();

		if (line.hasOption(ParentNestedConfigurableOptionSet.CONFIGSHORT)) {
			nestedConfig
					.optionValuesFromArgs(line
							.getOptionValues(ParentNestedConfigurableOptionSet.CONFIGSHORT));
		}

		configurableOptions.put(ParentNestedConfigurableOptionSet.CONFIGLONG,
				nestedConfig);

	}

	@SuppressWarnings("unchecked")
	private T checkCurrent(Enum<?> current) {
		EnumSet<T> valueSet = EnumSet.allOf(configurationOption.getClass());
		checkArgument(valueSet.contains(current), "%s isnt in enum set: %s",
				current, valueSet.toString());
		return (T) current;
	}

	public static <T extends Enum<T> & NestedConfigurableType<C, T>, C extends NestedConfigurable<C, T>, P extends ParentNestedConfigurable<T, C, P>> NestedConfiguration<T, C, P> buildNestedConfiguration(
			ParentNestedConfigurableOptionSet<T, C, P> options) {
		return new NestedConfiguration<T, C, P>(options);
	}
}
