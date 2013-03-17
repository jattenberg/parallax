/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Iterator;

import org.junit.Test;

import com.parallax.ml.util.ScaledNormalizing;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;

public class TestVectorNormalizingPipe {

	File file = new File("data/iris.data");

	@Test
	public void testVectorNormalizingPipe() {
		for (ScaledNormalizing vs : ScaledNormalizing.values()) {
			Pipeline<File, LinearVector> pipeline;
			pipeline = Pipeline.newPipeline(new FileSource(file))
					.addPipe(new FileToLinesPipe())
					.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4))
					.addPipe(new VectorNormalizingPipe(vs));
			
	        Iterator<Context<LinearVector>> out = pipeline.process();
	        assertTrue( out.hasNext() );
	        
	        while(out.hasNext()) {
	        	assertTrue(out.hasNext());
	        	Context<LinearVector> context = out.next();
	        	LinearVector lv = context.getData();
	        	assertTrue(lv.L0Norm() >= 0);
	        	assertTrue(lv.L1Norm() >= 0);
	        	assertTrue(lv.L2Norm() >= 0);
	        	assertTrue(lv.LInfinityNorm() >= 0);
	        }
	        
		}
		

	}

}
