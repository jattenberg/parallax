/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import org.apache.commons.cli.GnuParser;


/**
 * interface for classes that can be configured from a configurable object.
 *
 * @param <C> the concrete configurable type, used for method chaining
 * @author jattenberg
 */
public interface Configurable<C extends Configurable<C>>
{
	
	/**
	 * Returns a configuration that represents the current status of the object.
	 * 
	 * @return the configuration containing current options and their values.
	 */
	public abstract Configuration<C> getConfiguration();
	
	/**
	 * Populate a configuration so that represents the current status of the object.
	 * Typically used internally only, referenced by {@link #getConfiguration()}
	 *
	 * @param conf input configuration used to contain object info.
	 * @return the configuration containing current options and their values.
	 */
	public abstract Configuration<C> populateConfiguration(Configuration<C> conf);
	
	/**
	 * populate the variables of the configurable object using a gnu command line
	 * representation of the configuration. 
	 * 
	 * options are parsed internally into a configuration using {@link GnuParser#parse(org.apache.commons.cli.Options, String[])}
	 * then the configurable object is manipulated using {@link #configure(Configuration)}
	 *
	 * @see <a href="http://commons.apache.org/cli/">http://commons.apache.org/cli/</a> for info on args pasring
	 * @param args GNU cli representation of the configuration
	 * @throws Exception one of: IllegalArgumentException, SecurityException, ParseException, IllegalAccessException, InvocationTargetException, NoSuchMethodException
	 */
	public abstract void configureFromArguments(String[] args) throws Exception;
	
	/**
	 * Configure the internal variables of the object using the state of the configurable object
	 * using the information stored in input configuration. 
	 *
	 * @param configuration A configuration object containing the desired configuration
	 */
	public abstract void configure(Configuration<C> configuration);
	
	/**
	 * Prints a help message describing the configurable options and their bounds.
	 */
	public abstract void getHelp();
	
	/**
	 * Generates a configuration object representing the object's current configuration
	 * using {@link #getConfiguration()}, then transforms this configuration into a GNU
	 * cli string array representation. {@link #configureFromArguments(String[])} performs
	 * the inverse operation
	 * 
	 * @see <a href="http://commons.apache.org/cli/">http://commons.apache.org/cli/</a>
	 * @return the CLI arguments representing the current object configuration.
	 */
	public abstract String[] getArgumentsFromConfiguration();
}
