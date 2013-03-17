package com.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

import com.google.common.collect.Maps;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.projection.DataNormalization;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;

/**
 * The Class TestDataNormalizationPipe
 */
public class TestDataNormalizationPipe {
	
	/** The test data file */
	static String datadir = "/Users/spchopra/research/ml/dsi/data/";
	static String filename = datadir + "spambase.csv";
	static int inpDim = 57;
	static Map<String, String> labelMap = Maps.newHashMap();


	/**
	 * Function to test the normalization
	 * 
	 */
	@Test
	public void testNormalizationWorks() {
		// create the label map 
		labelMap.put("1", "" + 1);
		labelMap.put("0", "" + 0);

		DataNormalization dnorm = new DataNormalization(inpDim);

		Pipeline<File, BinaryClassificationInstance> pipeline;		
		pipeline = Pipeline
				.newPipeline(new FileSource(filename))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(",", inpDim, labelMap))
				.addPipe(new DataNormalizationPipe(dnorm))
				.addPipe(new BinaryInstancesFromVectorPipe(new BinaryTargetNumericParser()));
		
		assertTrue(!dnorm.isBuilt());
		Iterator<Context<BinaryClassificationInstance>> it = pipeline.process();
		
		// print the normalization parameters
		System.out.println("Normalization params:");
		System.out.println(dnorm);
		
		int ctr = 0;
		while (it.hasNext()) {
			BinaryClassificationInstance inst = it.next().getData();
			assertTrue(dnorm.isBuilt());
			if (ctr < 5) {
				inst.printInstance();
			}
			ctr++;
		}
	}
}
