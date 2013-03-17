/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.dictionary;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

import com.parallax.ml.util.lexer.Lexer;
import com.parallax.ml.vector.LinearVector;

/**
 * Dictionary is interface to define creation method of LinearVector
 *
 * @author Josh Attenberg
 * @see AbstractDictionary
 * @see BiMapReversableDictionary
 * @see HashDictionary
 * @see ReversableDictionary
 */
public interface Dictionary extends Serializable {
    public int getDimension();
    public LinearVector vectorFromText(String text, Lexer lexer);
    public LinearVector vectorFromText(Collection<String> text);
    public LinearVector vectorFromNamespacedText(Map<String, String> namespacedText, Lexer lexer);
    public LinearVector vectorFromNamespacedText(Map<String, Collection<String>> namespacedText);
    
    public LinearVector vectorFromNamespacedText(Map<String, String> namespacedText, Lexer lexer, boolean namespace);
    public LinearVector vectorFromNamespacedText(Map<String, Collection<String>> namespacedText, boolean namespace);
    
    public int dimensionFromString(String input);
    
    public void setBinaryFeatures(boolean binaryFeatures);

    
}
