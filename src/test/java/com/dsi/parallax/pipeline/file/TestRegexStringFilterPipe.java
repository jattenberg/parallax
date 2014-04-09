/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.file;

import static junit.framework.Assert.assertEquals;
import static junit.framework.Assert.assertNotSame;

import java.util.Iterator;

import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.text.RegexStringFilterPipe;
import com.google.common.collect.Iterators;

public class TestRegexStringFilterPipe {
    private static final String BASIC_TEST_VALUE = "0512-12345678";
    private static final String MATCH_REGEX_VALUE = "0512-\\d{8}";
    private static final String NOT_MATCH_REGEX_VALUE = "abc";

    @SuppressWarnings("deprecation")
	@Test
    public void test()
    {
        //Prepare the match regex object "0512-\d{8}"
        RegexStringFilterPipe regexStringFilterPipe = new RegexStringFilterPipe(MATCH_REGEX_VALUE);
        //Because the test data matches regex rule, it will return DummyIterator object
        assertNotSame(getActualString(regexStringFilterPipe), BASIC_TEST_VALUE);
        //Prepare the match regex object "abc"
        regexStringFilterPipe = new RegexStringFilterPipe(NOT_MATCH_REGEX_VALUE);
        //Because the test data doesn't match regex rule, it will return SingletonContextIterator object
        assertEquals(getActualString(regexStringFilterPipe), BASIC_TEST_VALUE);
    }

    /**
     * This method is to prepare test data (0512-12345678) and get actual data via running processIterator
     * @param regexStringFilterPipe RegexStringFilterPipe
     * @return actual string to be compared
     */
    private String getActualString(RegexStringFilterPipe regexStringFilterPipe)
    {
        String actual_string = null;
        Context<String> stringContext = new Context<String>(BASIC_TEST_VALUE);
        @SuppressWarnings("unchecked")
		Iterator<Context<String>> it = regexStringFilterPipe.processIterator(Iterators.forArray(stringContext));
        while (it.hasNext()) {
            actual_string = it.next().getData();
        }
        return actual_string;
    }
}
