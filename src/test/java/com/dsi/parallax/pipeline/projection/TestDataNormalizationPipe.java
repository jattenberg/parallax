package com.dsi.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.projection.DataNormalization;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.google.common.collect.Maps;

/**
 * The Class TestDataNormalizationPipe
 */
public class TestDataNormalizationPipe {
	
	/** The test data file */
	static String datadir = "data/";
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

		
		int ctr = 0;
		while (it.hasNext()) {
			BinaryClassificationInstance inst = it.next().getData();
			assertTrue(dnorm.isBuilt());
			ctr++;
		}
	}
}
