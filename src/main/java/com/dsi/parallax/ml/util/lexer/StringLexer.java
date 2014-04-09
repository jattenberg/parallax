/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.lexer;

import java.util.regex.Pattern;

/**
 * StringLexer
 *
 * @author Josh Attenberg
 */
public class StringLexer extends AbstractLexer {

    private static final long serialVersionUID = 4371206853023104175L;
    private String matchText;
    private boolean matchTextFresh;

    /**
     * Class constructor specifying string of lexer and regular expression to create
     * @param input string of lexer
     * @param regex regular expression
     */
    public StringLexer(String input, Pattern regex) {
        super(input, regex);
    }

    /**
     * Class constructor specifying string of lexer to create
     * @param input string of lexer
     */
    public StringLexer(String input) {
        super(input);
    }

    /**
     * Class constructor
     */
    public StringLexer() {
        super();
    }

    /**
     * Class constructor specifying regular expression to create
     * @param regex regular expression
     */
    public StringLexer(Pattern regex) {
        super(regex);
    }

    /**
     * The method updates Match text
     */
    private void updateMatchText ()
    {
        if (matcher != null && matcher.find()) {
            matchText = matcher.group();
            if (matchText.length() == 0) 
                updateMatchText();
        } else
            matchText = null;
        matchTextFresh = true;
    }

    /**
     * The method gets first offset
     * @return first offset
     */
    @Override
    public int getStartOffset() {
        if(null == matchText)
            return -1;
        return matcher.start();
    }

    /**
     * The method gets last offset
     * @return last offset
     */
    @Override
    public int getEndOffset() {
        if(null == matchText)
            return -1;
        return matcher.end();
    }

    /**
     * The method gets token string
     * @return match text
     */
    @Override
    public String getTokenString() {
        return matchText;
    }

    /**
     * The method checks if the StringLexer has next element
     * @return boolean
     */
    @Override
    public boolean hasNext() {
        if(!matchTextFresh)
            updateMatchText();
        return matchText != null;
    }

    /**
     * The method gets next string lexer element
     * @return StringLexer element
     */
    @Override
    public String next() {
        if(!matchTextFresh)
            updateMatchText();
        matchTextFresh = false;
        return matchText;
    }

    /**
     * The method adds string lexer element
     * @param input string of lexer
     */
    @Override
    public void addInput(String input) {
        super.addInput(input);
        this.matchText = null;
        this.matchTextFresh = false;
    }
}
