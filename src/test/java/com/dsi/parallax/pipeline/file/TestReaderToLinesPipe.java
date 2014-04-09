package com.dsi.parallax.pipeline.file;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.file.FileReaderSource;
import com.dsi.parallax.pipeline.file.ReaderToLinesPipe;

public class TestReaderToLinesPipe {

	File file = new File(".gitignore");

	@Test
	public void test() throws IOException {

		Pipeline<BufferedReader, String> pipeline;
		pipeline = Pipeline.newPipeline(new FileReaderSource(file)).addPipe(
				new ReaderToLinesPipe());

		Iterator<Context<String>> it = pipeline.process();
		assertTrue(it.hasNext());
		int count = 0;
		while (it.hasNext()) {
			it.next();
			count++;
		}
		assertTrue(!it.hasNext());

		int ct2 = 0;
		BufferedReader reader = new BufferedReader(new FileReader(file));
		@SuppressWarnings("unused")
		String line;
		while (null != (line = reader.readLine()))
			ct2++;
		assertEquals(ct2, count);
	}

}
