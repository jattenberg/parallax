package com.parallax.pipeline.text;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.file.FileToLinesPipe;

public class TestStringSplitPipe {

	File file = new File("./data/iris.data");
	int size = 5;

	@Test
	public void testParsesLines() {
		Pipeline<File, List<String>> pipeline = Pipeline
				.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new StringSplitPipe(','));
		Iterator<Context<List<String>>> it = pipeline.process();
		assertTrue(it.hasNext());

		while (it.hasNext()) {
			Context<List<String>> context = it.next();
			assertEquals(context.getData().size(), 5);
		}
	}

}
