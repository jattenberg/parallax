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
import com.dsi.parallax.pipeline.text.StringPatternReplacePipe;
import com.google.common.collect.Lists;

public class TestStringPatternReplacePipe {
    private static final String BASIC_TEST_VALUE = "TestData";// a part of test data
    private static final String REGEX_VALUE = "TestData\\d$"; // Regex value for test data
    private static final String REPLACE_VALUE = "ReplaceValue";

    @Test
    public void test()
    {
        //Get these test data
        List<Context<String>> contextList = generateStringReplace();
        Iterator<Context<String>> it = contextList.iterator();

        StringPatternReplacePipe pipe = new StringPatternReplacePipe(REGEX_VALUE,REPLACE_VALUE);
        Iterator<Context<String>> out = pipe.processIterator(it);
        assertTrue(out.hasNext());

        int ct = 0;
        while(out.hasNext()) {
            String actual_result = out.next().getData();
            ++ct;
            assertEquals(actual_result,REPLACE_VALUE);
        }
        assertEquals(ct,contextList.size());

    }

    /**
     * This method is to generate test data of "String Replace"
     * @return List of Context<String> type
     */
    private List<Context<String>> generateStringReplace()
    {
        List<Context<String>> contextList =  Lists.newArrayList();
        for(int i=0;i<10;i++)
        {
            contextList.add(new Context<String>(BASIC_TEST_VALUE+i));
        }
        return contextList;
    }
}
