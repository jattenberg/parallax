/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.examples.pipeline;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.text.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * This class is to test simple pipeline combination
 */
public class DataExperiment {
    //define the test data file
    static File file = new File(
            "data/adult.small.vw");
    //define the test data
	static int bins = (int) Math.pow(2, 16);

    /**
     * This method is to test simple pipeline
     * @param args it defines nothing
     * @throws IOException because it reads file, it will throw io exception
     */
	public static void main(String[] args) throws IOException {
		Pipeline<File, LinearVector> pipeline;
		pipeline = Pipeline.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
                .addPipe(new StringToVWPipe())
				.addPipe(new VWtoLabeledStringPipe())
				.addPipe(new TextSanitizerPipe())
				.addPipe(new StringToTokenSequencePipe())
				.addPipe(new StringSequenceToNGramsPipe(new int[] { 2 }))
				.addPipe(new StringSequenceToNumericDictionaryPipe(bins));
		
		Iterator<Context<LinearVector>> it = pipeline.process();
		
        int ct = 0;
        while (it.hasNext()) {
            ++ct;
            LinearVector inst = it.next().getData();
            assertEquals(inst.size(), bins);
            assertTrue(inst.L0Norm() > 0);
            assertTrue(inst.L1Norm() > 0);
            assertTrue(inst.L2Norm() > 0);
        }
        int ct2 = 0;
        BufferedReader reader = new BufferedReader(new FileReader(file));
        @SuppressWarnings("unused")
        String line;
        while (null != (line = reader.readLine()))
            ct2++;
        assertEquals(ct2, ct);
	}

}
