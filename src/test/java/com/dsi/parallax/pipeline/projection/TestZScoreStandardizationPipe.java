package com.dsi.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.csv.NumericCSVtoLabeledVectorPipe;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.instance.BinaryClassificationInstancesSink;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.dsi.parallax.pipeline.projection.ZScoreStandardizationPipe;
import com.dsi.parallax.pipeline.text.StringSequenceToNGramsPipe;
import com.dsi.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.dsi.parallax.pipeline.text.StringToTokenSequencePipe;
import com.dsi.parallax.pipeline.text.TextSanitizerPipe;
import com.google.common.collect.Maps;

public class TestZScoreStandardizationPipe {

	/** The file. */
	File file = new File(".classpath");

	/** The bins. */
	int bins = 100;

	/**
	 * Test projection works.
	 * 
	 * @throws IOException
	 *             Signals that an I/O exception has occurred.
	 */
	@Test
	public void testProjectionWorks() throws IOException {

		Pipeline<File, BinaryClassificationInstance> pipeline;

		ZScoreStandardizationPipe zsc = new ZScoreStandardizationPipe();

		pipeline = Pipeline.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new TextSanitizerPipe())
				.addPipe(new StringToTokenSequencePipe())
				.addPipe(new StringSequenceToNGramsPipe(2))
				.addPipe(new StringSequenceToNumericDictionaryPipe(bins))
				.addPipe(zsc).addPipe(new BinaryInstancesFromVectorPipe());

		@SuppressWarnings("unused")
		Iterator<Context<BinaryClassificationInstance>> it = pipeline.process();
		assertTrue(zsc.isTrained());

	}
	
	@Test
	public void testOnIris() {
		Pipeline<File, BinaryClassificationInstance> pipeline;
		Map<String, String> labelMap = Maps.newHashMap();
		labelMap.put("Iris-setosa", 1 + "");
		labelMap.put("Iris-versicolor", 0 + "");
		labelMap.put("Iris-virginica", 0 + "");

		ZScoreStandardizationPipe zsc = new ZScoreStandardizationPipe();
		
		pipeline = Pipeline
				.newPipeline(new FileSource(new File("data/iris.data")))
				.addPipe(new FileToLinesPipe())
				.addPipe(new NumericCSVtoLabeledVectorPipe(-1, 4, labelMap))
				.addPipe(zsc)
				.addPipe(
						new BinaryInstancesFromVectorPipe(
								new BinaryTargetNumericParser()));
		BinaryClassificationInstancesSink sink = new BinaryClassificationInstancesSink();
		sink.setSource(pipeline);
		assertTrue(zsc.isTrained());
		
	}

}
