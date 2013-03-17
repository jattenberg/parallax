/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.lexer;

import java.util.Iterator;

/**
 * class for constructing token sequences from strings
 *
 * @author jattenberg
 */
public interface Lexer extends Iterator<String> {
    
    public void addInput(String input);
    
    public int getStartOffset ();

    public int getEndOffset ();

    public String getTokenString ();

    public boolean hasNext ();

    // Returns token text as a String
    public String next ();

    public void remove ();
}
