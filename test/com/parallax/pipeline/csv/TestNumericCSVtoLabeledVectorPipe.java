/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.csv;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;

public class TestNumericCSVtoLabeledVectorPipe {

	File file = new File("data/iris.data");
	@Test
	public void testReadsFile() {
		Pipeline<File, LinearVector> pipeline;
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe())
                .addPipe(new NumericCSVtoLabeledVectorPipe(-1,4));
        Iterator<Context<LinearVector>> out = pipeline.process();
        assertTrue( out.hasNext() );
        
        while(out.hasNext()) {
        	assertTrue(out.hasNext());
        	Context<LinearVector> context = out.next();
        	assertTrue(null != context.getLabel());
        	assertTrue(null == context.getId());
        	assertTrue(context.getData().size() == 4);
        }
	}
	
	@Test
	public void testCorrectLabel() {
		Pipeline<File, LinearVector> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1+"");
		labelMap.put("Iris-versicolor", 0+"");
		labelMap.put("Iris-virginica", 0+"");
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe())
                .addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap));
        Iterator<Context<LinearVector>> out = pipeline.process();
        assertTrue( out.hasNext() );
        int total = 0;
        while(out.hasNext()) {
        	assertTrue(out.hasNext());
        	Context<LinearVector> context = out.next();
        	assertTrue(null != context.getLabel());
        	total += Integer.parseInt(context.getLabel());
        	assertTrue(null == context.getId());
        	assertTrue(context.getData().size() == 4);
        }
        assertEquals(50, total);
	}

}
