/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.text;

import static junit.framework.Assert.assertEquals;

import java.util.Iterator;

import org.junit.Test;

import com.google.common.collect.Iterators;
import com.parallax.ml.util.VW;
import com.parallax.pipeline.Context;

public class TestVWtoNamespaceTextPipe {
    private static final String EXPECT_KEY = "Text";
    private static final String EXPECT_VALUE = "Expect_Value";
    @Test
    public void test()
    {
        //Create test data : VW and Context
        VW vw = new VW();
        vw.addData(EXPECT_KEY,EXPECT_VALUE);
        Context<VW> context = new Context<VW>(vw);
        //Invoke VWtoNamespaceTextPipe 's  processIterator
        VWtoNamespaceTextPipe vWtoNamespaceTextPipe = new VWtoNamespaceTextPipe();
        @SuppressWarnings("unchecked")
		Iterator<Context<NamespaceText>> it =  vWtoNamespaceTextPipe.processIterator(Iterators.forArray(context));

        NamespaceText namespaceText = null;
        while (it.hasNext()) {
            namespaceText = it.next().getData();
        }

        if (namespaceText != null) {
            assertEquals(EXPECT_VALUE, namespaceText.getTextForNamespace(EXPECT_KEY));
        }
    }
}
