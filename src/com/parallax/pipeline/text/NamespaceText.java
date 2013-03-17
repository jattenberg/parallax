/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.text;

import java.util.Map;

import com.google.common.collect.Maps;
import com.parallax.ml.util.VW;

/**
 * NamespaceText is JavaBean stores contents of VM
 *
 * @author Josh Attenberg
 */
public class NamespaceText {
    private final Map<String, String> namespaceText;

    /**
     * Class constructor
     */
    public NamespaceText() {
        namespaceText = Maps.newHashMap();
    }

    /**
     * Class constructor specifying VW of objects to create.
     * @param vw VW
     */
    public NamespaceText(VW vw) {
        namespaceText = Maps.newHashMap(vw.getNamespaceData());
    }

    /**
     * Class constructor specifying namespace text to create.
     * @param namespaceText namespace text
     */
    public NamespaceText(Map<String, String> namespaceText) {
        this.namespaceText = Maps.newHashMap(namespaceText);
    }

    /**
     * The method gets namespaceText by namespace.
     * @param namespace namespace
     * @return namespaceText
     */
    public String getTextForNamespace(String namespace) {
        return namespaceText.get(namespace);
    }

    /**
     * The method checks if the namespace exists in the namespaceText
     * @param namespace namespace
     * @return boolean
     */
    public boolean containsNamespace(String namespace) {
        return namespaceText.containsKey(namespace);
    }   
}
