/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.dictionary;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.parallax.ml.util.lexer.Lexer;
import com.parallax.ml.vector.LinearVector;

/**
 * AbstractDictionary is abstract class and implements Dictionary Interface
 *
 * @author Josh Attenberg
 */
public abstract class AbstractDictionary implements Dictionary {

    private static final long serialVersionUID = 5202830599228901827L;
    protected final int dimension;
    protected boolean binaryFeatures = false;

    protected AbstractDictionary(int dimension) {
        this.dimension = dimension;
    }

    protected AbstractDictionary(int dimension, boolean binaryFeatures) {
        this.dimension = dimension;
        this.binaryFeatures = binaryFeatures;
    }

    /**
     * The method gets dimension
     * @return int dimension
     */
    @Override
    public int getDimension() {
        return dimension;
    }

    /**
     * The method sets binary features
     * @param binaryFeatures binary features
     */
    @Override
    public void setBinaryFeatures(boolean binaryFeatures) {
        this.binaryFeatures = binaryFeatures;
    }

    /**
     * The method create LinearVector by namespace text and lexer
     * @param namespacedText namespace text
     * @param lexer Lexer object
     * @return LinearVector
     */
    @Override
    public LinearVector vectorFromNamespacedText(
            Map<String, String> namespacedText, Lexer lexer) {
        return vectorFromNamespacedText(namespacedText, lexer, true);
    }

    /**
     * The method creates LinerVector by namespace text
     * @param namespacedText namespace text
     * @return LinearVector
     */
    public LinearVector vectorFromNamespacedText(
            Map<String, Collection<String>> namespacedText) {
        return vectorFromNamespacedText(namespacedText, true);
    }

    /**
     * The method creates LinearVector by multiple namespace text, lexer
     * @param namespacedText multiple namespace text
     * @param lexer lexer
     * @param namespace namespace
     * @return LinearVector
     */
    @Override
    public LinearVector vectorFromNamespacedText(
            Map<String, String> namespacedText, Lexer lexer, boolean namespace) {
        Map<String, Collection<String>> tmp = Maps.newHashMap();
        for(String key : namespacedText.keySet()) {
            lexer.addInput(namespacedText.get(key));
            List<String> tokens = Lists.newArrayList(lexer);
            tmp.put(key, tokens);
        }
        return vectorFromNamespacedText(tmp, namespace);
    }

    /**
     * The method creates LinearVector by text and lexer
     * @param text string text
     * @param lexer lexer
     * @return LinearVector
     */
    @Override
    public LinearVector vectorFromText(String text, Lexer lexer) {
        lexer.addInput(text);
        return vectorFromText(Lists.newArrayList(lexer));
    }
}
