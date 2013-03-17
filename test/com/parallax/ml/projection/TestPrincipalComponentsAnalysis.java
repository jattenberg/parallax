/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.projection;

import static org.junit.Assert.assertTrue;

import java.util.Collection;

import org.junit.Test;

import com.google.common.collect.Lists;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;

/**
 * The Class TestPrincipalComponentsAnalysis.
 */
public class TestPrincipalComponentsAnalysis {

    /**
     * Test pca works.
     */
    @Test
    public void testPCAWorks() {
        Collection<LinearVector> X = Lists.newArrayList();
        for(int i = 0; i < 5; i++) {
            LinearVector x = LinearVectorFactory.getVector(3);
            for(int j = 0; j < 3; j++)
                x.updateValue(j, i*j);

            X.add(x);
        }
        
        PrincipalComponentsAnalysis pca = new PrincipalComponentsAnalysis(3, 2, X);
        for(LinearVector x : X) {
        	assertTrue(pca.errorMembership(x.getW()) < 2);
        }
            
    }

}
