/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.text.StringSequencePatternReplacePipe;
import com.google.common.collect.Lists;

public class TestStringSequencePatternReplacePipe {
    private static final String REGEX_VALUE = "\\w"; // Regex value for test data
    private static final String REPLACE_VALUE = "ReplaceValue";

    @Test
    public void test()
    {
        List<String> a = Lists.newArrayList("a");
        List<String> b = Lists.newArrayList("b");
        List<String> c = Lists.newArrayList("c");

        @SuppressWarnings("unchecked")
        List<Context<List<String>>> comb = Lists.newArrayList(new Context<List<String>>(a),
                new Context<List<String>>(b),new Context<List<String>>(c));

        Iterator<Context<List<String>>> it = comb.iterator();

        StringSequencePatternReplacePipe pipe = new StringSequencePatternReplacePipe(REGEX_VALUE,REPLACE_VALUE);
        Iterator<Context<List<String>>> contextIterator = pipe.processIterator(it);

        assertTrue(contextIterator.hasNext());
        int ct = 0;
        while(contextIterator.hasNext()) {
            List<String> list = contextIterator.next().getData();
            ++ct;
            assertEquals(list.get(0),REPLACE_VALUE);
        }
        assertEquals(ct,comb.size());
    }
}
