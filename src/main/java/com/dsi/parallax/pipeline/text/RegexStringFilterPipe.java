/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.regex.Pattern;

import com.dsi.parallax.pipeline.AbstractFilteringPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

/**
 * RegexStringFilterPipe is primitive class for filtering contents of String object
 * via regular expression
 *
 * @author Josh Attenberg
 */
public class RegexStringFilterPipe extends AbstractFilteringPipe<String> {

	private static final long serialVersionUID = -8280535935539549965L;
    private Pattern regex;

    /**
     * Class constructor specifying regular expression to create
     * @param toRemoveRegex string of regular expression pattern
     */
    public RegexStringFilterPipe(String toRemoveRegex){
    	this(Pattern.compile(toRemoveRegex));
    }

    /**
     * Class constructor specifying regular expression pattern to create
     * @param regex regular expression pattern
     */
	public RegexStringFilterPipe(Pattern regex) {
		super();
		this.regex = regex;
	}

    /**
     * The method returns the class's Type "RegexStringFilterPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<RegexStringFilterPipe>(){}.getType();
	}

	@Override
	protected boolean operate(Context<String> context) {
		return !regex.matcher(context.getData()).find();
	}

}
