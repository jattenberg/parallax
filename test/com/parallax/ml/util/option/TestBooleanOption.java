/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.option;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class TestBooleanOption {
    @Test
    public void testCopyOption(){
        BooleanOption booleanOption = getInstance();
        BooleanOption copyOptionObject = (BooleanOption)booleanOption.copyOption();
        assertTrue(booleanOption.getDEFAULTB() == copyOptionObject.getDEFAULTB());
        assertEquals(booleanOption.getDescription(),copyOptionObject.getDescription());
        assertEquals(booleanOption.getLongName(),copyOptionObject.getLongName());
        assertEquals(booleanOption.getShortName(),copyOptionObject.getShortName());
    }

    private BooleanOption getInstance(){
        String shortName = "test_shortName";
        String longName = "test_longName";
        String desc = "test_desc";
        boolean defaultv = true;
        boolean optimizable = false ;
        return new BooleanOption(shortName,longName,desc,defaultv,optimizable);
    }
}
