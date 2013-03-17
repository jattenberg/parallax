/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.pipeline.instance;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.junit.Test;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.target.BinaryTargetNumericParser;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.instance.BinaryClassificationInstanceSink;
import com.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;
import com.parallax.pipeline.text.StringSequenceToNGramsPipe;
import com.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.parallax.pipeline.text.StringToTokenSequencePipe;
import com.parallax.pipeline.text.TextSanitizerPipe;


/**
 * The Class TestInstanzeSink.
 */
public class TestInstanzeSink {

    /** The file. */
    File file = new File(".classpath");
    
    /** The bins. */
    int bins = 10000;

    /**
     * Test sinking.
     *
     * @throws IOException Signals that an I/O exception has occurred.
     */
    @Test
    public void testSinking() throws IOException {
        BinaryClassificationInstanceSink sink = new BinaryClassificationInstanceSink();
        Pipeline<File, BinaryClassificationInstance> pipeline;
        pipeline = Pipeline.newPipeline(new FileSource(file))
                .addPipe(new FileToLinesPipe())
                .addPipe(new TextSanitizerPipe())
                .addPipe(new StringToTokenSequencePipe())
                .addPipe(new StringSequenceToNGramsPipe(2))
                .addPipe(new StringSequenceToNumericDictionaryPipe(bins))
                .addPipe(new BinaryInstancesFromVectorPipe(new BinaryTargetNumericParser()));
        
        assertTrue(!sink.hasNext() );
        sink.setSource(pipeline);
        assertTrue( sink.hasNext() );
        
        int ct = 0;
        while (sink.hasNext()) {
            ++ct;
            assertTrue(sink.hasNext());
            @SuppressWarnings("unused")
            BinaryClassificationInstance inst = sink.next();
        }
        int ct2 = 0;
        BufferedReader reader = new BufferedReader(new FileReader(file));
        @SuppressWarnings("unused")
        String line;
        while (null != (line = reader.readLine()))
            ct2++;
        assertEquals(ct2, ct);
        assertTrue(!sink.hasNext());
       
    }
}
