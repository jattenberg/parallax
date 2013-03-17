/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.io.File;

import org.junit.Test;

import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.text.StringSequenceToNGramsPipe;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.TextSanitizerPipe;

public class TestPipelineSerializer {
	
    File file = new File("README");
    int bins = 10000;
    
	@Test
	public void test() {
        @SuppressWarnings("unused")
		Pipeline<File, LinearVector> pipeline;
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe())
                .addPipe(new TextSanitizerPipe())
                .addPipe(new StringToTokenSequencePipe())
                .addPipe(new StringSequenceToNGramsPipe(2))
                .addPipe(new StringSequenceToNumericDictionaryPipe(bins));
	}

}
