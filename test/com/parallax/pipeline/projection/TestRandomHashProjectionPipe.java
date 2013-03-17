/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.projection;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import org.junit.Test;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.text.StringSequenceToNGramsPipe;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.TextSanitizerPipe;

/**
 * The Class TestRandomHashProjectionPipe.
 */
public class TestRandomHashProjectionPipe {

    /** The file. */
    File file = new File(".classpath");
    
    /** The bins. */
    int bins = 10000;

    /**
     * Test projection works.
     *
     * @throws IOException Signals that an I/O exception has occurred.
     */
    @Test
    public void testProjectionWorks() throws IOException {

        for (int i = 1; i < 100; i++) {
            Pipeline<File, BinaryClassificationInstance> pipeline;
            pipeline = Pipeline.newPipeline(new FileSource(file))
                    .addPipe(new FileToLinesPipe())
                    .addPipe(new TextSanitizerPipe())
                    .addPipe(new StringToTokenSequencePipe())
                    .addPipe(new StringSequenceToNGramsPipe(2))
                    .addPipe(new StringSequenceToNumericDictionaryPipe(bins))
                    .addPipe(new RandomHashProjectionPipe(bins, i))
                    .addPipe(new BinaryInstancesFromVectorPipe());

            Iterator<Context<BinaryClassificationInstance>> it = pipeline.process();
            
            while (it.hasNext()) {
            	BinaryClassificationInstance inst = it.next().getData();
                assertTrue(inst.L0Norm() <= i);
            }

        }
    }
}
