/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.file;

import static junit.framework.Assert.assertEquals;

import java.util.Iterator;

import org.junit.Test;

import com.google.common.collect.Iterators;
import com.parallax.ml.util.VW;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.text.NamespaceStringSelectorPipe;
import com.parallax.pipeline.text.NamespaceText;

/**
 * Prepare one test data (Text = Expect_Value) to test the namespace.
 *
 * @author esteban
 *
 */
public class TestNamespaceStringSelectorPipe {
    private static final String EXPECT_KEY = "Text";
    private static final String EXPECT_VALUE = "Expect_Value";

    @Test
    public void test(){
        //Prepare test data
        Context<NamespaceText> context = generateContextNamespace();

        NamespaceStringSelectorPipe namespaceStringSelectorPipe = new NamespaceStringSelectorPipe(EXPECT_KEY);
        @SuppressWarnings("unchecked")
		Iterator<Context<String>> it = namespaceStringSelectorPipe.processIterator(Iterators.forArray(context));
        String namespaceText = null;
        while (it.hasNext()) {
            namespaceText = it.next().getData();
        }

        assertEquals(EXPECT_VALUE, namespaceText);
    }

    /**
     * This method is to generate a simple Context<Namespace> object
     * @return Context<Namespace>
     */
    private Context<NamespaceText> generateContextNamespace(){
        //Create an VW object and set test data Text = Expect_Value
        VW vw = new VW();
        vw.addData(EXPECT_KEY,EXPECT_VALUE);
        //NamespaceText object and set VW to it
        NamespaceText namespaceText = new NamespaceText(vw);
        return new  Context<NamespaceText>(namespaceText);
    }
}
