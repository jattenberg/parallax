/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.option;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class TestEnumOption {


    @Test
    public void testCheckCurrent(){
        EnumOption<EnumTest> enumOption = getInstance();
        //test for positive
        assertEquals(EnumTest.TEST1,enumOption.checkCurrent(EnumTest.TEST1));
    }

    @Test
    public void testCopyOption(){
        EnumOption<EnumTest> enumOption = getInstance();
        @SuppressWarnings("unchecked")
		EnumOption<EnumTest> copyOptionObject = (EnumOption<EnumTest>)enumOption.copyOption();
        assertEquals(enumOption.getDescription(),copyOptionObject.getDescription());
        assertEquals(enumOption.getLongName(),copyOptionObject.getLongName());
        assertEquals(enumOption.getShortName(),copyOptionObject.getShortName());
        assertEquals(enumOption.getDEFAULTE(),EnumTest.TEST1);
        assertTrue(enumOption.getValueSet().contains(EnumTest.TEST2));

    }

    private EnumOption<EnumTest> getInstance(){
        String shortName = "test_shortName";
        String longName = "test_longName";
        String desc = "test_desc";
        boolean optimizable = false ;

        return new EnumOption<EnumTest>(shortName,longName,optimizable,desc,EnumTest.class,EnumTest.TEST1);
    }

    public enum EnumTest{
        TEST1,TEST2
    }

    public enum EnumA{
        TEST
    }
}
