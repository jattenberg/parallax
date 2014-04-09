/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

/**
 * StringSequencePatternReplacePipe is primitive class for finding contents of String via regular expression pattern
 * And getting List contents of replaced String object into context
 *
 * @author Josh Attenberg
 */
public class StringSequencePatternReplacePipe extends AbstractPipe<List<String>, List<String>>{

    private static final long serialVersionUID = 5250877845290090311L;
    private Pattern pattern;
    private String replacement;

    /**
     * Class constructor specifying regular expression pattern and replacement string of objects to create.
     * @param pattern regular expression pattern
     * @param replacement string of replacement
     */
    public StringSequencePatternReplacePipe(Pattern pattern, String replacement) {
        super();
    	this.pattern = pattern;
        this.replacement = replacement;
    }

    /**
     * Class constructor specifying regular expression string and replacement string of objects to create.
     * @param pattern regular expression string
     * @param replacement replacement string
     */
    public StringSequencePatternReplacePipe(String pattern, String replacement) {
        this(Pattern.compile(pattern), replacement);
    }

    /**
     * The method returns the class's Type "StringSequencePatternReplacePipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequencePatternReplacePipe>(){}.getType();
	}

	@Override
	protected Context<List<String>> operate(Context<List<String>> context) {
        List<String> payload = context.getData();
        List<String> out = Lists.newLinkedList();
        for(String part : payload) {
            Matcher m = pattern.matcher(part);
            String replaced = m.replaceAll(replacement);
            out.add(replaced);
        }
        return Context.createContext(context, out);
	}
}
