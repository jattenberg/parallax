package com.parallax.pipeline.csv;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import com.google.common.collect.Lists;
import com.parallax.ml.util.csv.CSV;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.Context;

public class TestCSVSource {

	File file = new File("./data/iris.data");

	@Test
	public void test() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		reader.mark(10000);
		String[] parts = reader.readLine().split(",");
		int size = parts.length;
		reader.reset();

		CSV csv = CSV.numericColumnCSV(size).setLabelColumn(size - 1)
				.initialize();

		CSVSource testSoure = new CSVSource(file, CSV.numericColumnCSV(size)
				.setLabelColumn(size - 1).initialize(), ',');

		List<Context<LinearVector>> testVecs = Lists.newArrayList(testSoure
				.provideData());
		List<Context<LinearVector>> worksVecs = Lists.newArrayList();

		String line;
		while (null != (line = reader.readLine())) {
			List<String> input = Arrays.asList(line.split(","));
			worksVecs.add(csv.parseRow(input));
		}

		assertEquals(testVecs.size(), worksVecs.size());

		for (int i = 0; i < worksVecs.size(); i++) {
			testCSVs(testVecs.get(i), worksVecs.get(i));
		}
	}

	protected void testCSVs(Context<LinearVector> works,
			Context<LinearVector> test) {
		assertNull(works.getId());
		assertNull(test.getId());

		assertNotNull(works.getLabel());
		assertNotNull(test.getLabel());
		assertTrue(works.getLabel().equals(test.getLabel()));

		assertEquals(works.getData().size(), test.getData().size());
		for (int i = 0; i < works.getData().size(); i++) {
			assertEquals(works.getData().getValue(i), test.getData()
					.getValue(i), 0);
		}
	}
}
