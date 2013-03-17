package com.parallax.ml.projection;

import java.util.Collection;

import org.junit.Test;

import com.google.common.collect.Lists;
import com.parallax.ml.vector.LinearVector;
import com.parallax.ml.vector.LinearVectorFactory;

/**
 * The Class TestDataNormailzation
 */
public class TestDataNormalization {

	    /**
	     * Test that the data normalization works
	     */
	    @Test
	    public void testDataNormalizationWorks() {
	    	// create a collection of dummy vectors
	        Collection<LinearVector> X = Lists.newArrayList();
	        for(int i = 0; i < 5; i++) {
	            LinearVector x = LinearVectorFactory.getVector(3);
	            for(int j = 0; j < 3; j++) {
	                x.updateValue(j, i*j);
	            }	
	            X.add(x);
	        }
	        
	        System.out.println("Data: ");
	        printData(X);	        
	        DataNormalization dnorm = new DataNormalization(3, X);
	        System.out.println(dnorm);
	        
	        // now try projecting the data vectors using the learnt norm
	        Collection<LinearVector> Y = Lists.newArrayList();
	        for(LinearVector x : X) {
	        	LinearVector output = dnorm.project(x);
	        	Y.add(output);	        	
	        }
	        System.out.println("Normalized Data:");
	        printData(Y);	            
	    }
	    
	    private void printData(Collection<LinearVector> data) {
	        for(LinearVector x : data) {
	        	for (int i = 0; i < x.size(); i++) { 
	        		System.out.print(x.getValue(i) + " ");
	        	}	
	        	System.out.println("");
	        }
	    }

	}
