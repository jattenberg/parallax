/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * StringPatternReplacePipe is primitive class for replace contents of String object into context
 * via regular pattern
 *
 * @author Josh Attenberg
 */
public class StringPatternReplacePipe extends AbstractPipe<String, String> {

	private static final long serialVersionUID = 5250877845290090311L;
	private Pattern pattern;
	private String replacement;

    /**
     * Class constructor specifying regular expression pattern and replacement string to create
     * @param pattern regular expression pattern
     * @param replacement replacement string
     */
	public StringPatternReplacePipe(Pattern pattern, String replacement) {
		super();
		this.pattern = pattern;
		this.replacement = replacement;
	}

    /**
     * Class constructor specifying regular expression string and replacement string to create
     * @param pattern string of regular expression
     * @param replacement string of replacement
     */
	public StringPatternReplacePipe(String pattern, String replacement) {
		this(Pattern.compile(pattern), replacement);
	}

    /**
     * The method returns the class's Type "StringPatternReplacePipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringPatternReplacePipe>() {
		}.getType();
	}

	@Override
	protected Context<String> operate(Context<String> context) {
		String line = context.getData();
		Matcher m = pattern.matcher(line);
		String replaced = m.replaceAll(replacement);
		return Context.createContext(context, replaced);
	}

}
