/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.option;

import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;

import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.ParseException;

/**
 * AbstractConfigurable is abstract class that can be configured from a configurable object.
 *
 * @author jattenberg
 */
public abstract class AbstractConfigurable<C extends Configurable<C>> implements Configurable<C>, Serializable
{
	private static final long serialVersionUID = -6345930160333350925L;
	protected static final double BIGVAL = Math.pow(2, 30);

	
	/**
	 * constructs a usage message for the subclasses arguments
	 */
	@Override
	public void getHelp()
	{
		System.err.println(getClass().getName());
		HelpFormatter formatter = new HelpFormatter();
		formatter.printHelp( getClass().getName(), getConfiguration().getCliOpts());
	}

    /**
     * The method gets Configuration and sets all these arguments of system to Configuration
     * @param args arguments of system
     * @throws IllegalArgumentException
     * @throws SecurityException
     * @throws ParseException
     * @throws IllegalAccessException
     * @throws InvocationTargetException
     * @throws NoSuchMethodException
     */
	@Override
	public void configureFromArguments(String[] args) throws IllegalArgumentException, SecurityException, ParseException, IllegalAccessException, InvocationTargetException, NoSuchMethodException {
		Configuration<C> conf = getConfiguration();
		conf.optionValuesFromArgs(args);
		configure(conf);
	}

    /**
     * The method gets arguments by invoking Configuration's getArgumentsFromOpts
     * @return argument array
     */
	@Override
	public String[] getArgumentsFromConfiguration() {
		Configuration<C> conf = getConfiguration();
		return conf.getArgumentsFromOpts();
	}
}
