/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import static org.junit.Assert.assertEquals;

import java.util.Collection;
import java.util.Iterator;

import org.junit.Test;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanValueBound;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.Option;
import com.dsi.parallax.ml.util.option.OptionSet;
@SuppressWarnings("rawtypes")
public class TestOptionSet {
    
	@Test
    public void testAddOption(){
        Option option = getFloatOption();
        OptionSet<?> optionSet = new OptionSet();
        optionSet.addOption(option);
        Collection<Option> co =  optionSet.getOptions();
        for (Option aCo : co) {
            assertEquals(aCo, option);
        }
    }

    @Test
    public void testIterator(){
        Option option = getFloatOption();
        OptionSet optionSet = new OptionSet();
        optionSet.addOption(option);
        Iterator it = optionSet.iterator();
        while (it.hasNext()){
            assertEquals(it.next(), option);
        }
    }

    @Test
    public void testCopyOptionSet(){
        Option option = getFloatOption();
        OptionSet optionSet = new OptionSet();
        optionSet.addOption(option);

        OptionSet newOptionSet = optionSet.copyOptionSet();
        assertEquals(newOptionSet.getOptions().size(),optionSet.getOptions().size());

    }

    private Option getFloatOption(){
        String shortName = "test_shortName_Float";
        String longName = "test_longName_Float";
        String desc = "test_desc";
        double min = 1;
        double max = 10;
        double defaultv = 5;
        boolean optimizable = false ;

        return new FloatOption(shortName, longName, desc, defaultv, optimizable, new GreaterThanValueBound(min), new LessThanValueBound(max));
    }
}
