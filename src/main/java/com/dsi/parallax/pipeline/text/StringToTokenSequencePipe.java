/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.List;
import java.util.regex.Pattern;

import com.dsi.parallax.ml.util.lexer.Lexer;
import com.dsi.parallax.ml.util.lexer.StringLexer;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

/**
 * turns input strings into ordered sequences of tokens
 *
 * @author jattenberg
 */
public class StringToTokenSequencePipe extends AbstractPipe<String, List<String>> {

    private static final long serialVersionUID = -2208971330459989832L;
    private final Lexer lexer;

    /**
     * Class constructor
     */
    public StringToTokenSequencePipe() {
    	super();
        lexer = new StringLexer();
    }

    /**
     * Class constructor specifying regular expression pattern to create
     * @param pattern regular expression pattern
     */
    public StringToTokenSequencePipe(Pattern pattern) {
    	super();
        lexer = new StringLexer(pattern);
    }

    /**
     * Class constructor specifying regular expression string to create
     * @param pattern regular expression string
     */
    public StringToTokenSequencePipe(String pattern) {
    	super();
        lexer = new StringLexer(Pattern.compile(pattern));
    }

    /**
     * The method returns the class's Type "StringToTokenSequencePipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringToTokenSequencePipe>(){}.getType();
	}

	@Override
	protected Context<List<String>> operate(Context<String> context) {
        String payload = context.getData();
        lexer.addInput(payload);
        
        List<String> output = Lists.newArrayList(lexer);
        return Context.createContext(context, output);
	}
}
