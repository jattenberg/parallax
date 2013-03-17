/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.lexer;

import java.io.Serializable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * AbstractLexer is abstract class that
 *
 * @author Josh Attenberg
 */
public abstract class AbstractLexer implements Lexer, Serializable {

    private static final long serialVersionUID = 695128812250634490L;

    public static final Pattern LEX_ALPHA = Pattern.compile("\\p{Alpha}+");
    public static final Pattern LEX_WORDS = Pattern.compile("\\w+");
    public static final Pattern LEX_NONWHITESPACE_TOGETHER = Pattern
            .compile("\\S+");
    public static final Pattern LEX_WORD_CLASSES = Pattern
            .compile("\\p{Alpha}+|\\p{Digit}+");
    public static final Pattern LEX_NONWHITESPACE_CLASSES = Pattern
            .compile("\\p{Alpha}+|\\p{Digit}+|\\p{Punct}");
    // Lowercase letters and uppercase letters
    public static final Pattern UNICODE_LETTERS = Pattern
            .compile("[\\p{Ll}&&\\p{Lu}]+");
    
    protected CharSequence input;
    protected Pattern regex;
    protected Matcher matcher = null;

    /**
     * Class constructor specifying string of lexer to create
     * @param input string of lexer
     */
    protected AbstractLexer(String input) {
        addInput(input);
        regex = LEX_ALPHA;
    }

    /**
     * Class constructor specifying string of lexer and regular expression to create
     * @param input string of lexer
     * @param regex regular expression
     */
    protected AbstractLexer(String input, Pattern regex) {
        this(input);
        this.regex = regex;
    }

    /**
     *  Class constructor specifying regular expression to create
      * @param regex regular expression
     */
    protected AbstractLexer(Pattern regex) {
        this.regex = regex;
    }
    
    protected AbstractLexer() {
        this.regex = LEX_ALPHA;
    }

    /**
     * The method adds string of lexer to CharSequence
     * @param input string of lexer
     */
    @Override
    public void addInput(String input) {
        this.input = input;
        if (input != null)
            this.matcher = regex.matcher(input);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("remove isnt supported.");
    }
}
