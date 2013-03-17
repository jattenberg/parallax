package com.parallax.ml.util.csv;

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
import java.util.Set;

import org.junit.Test;

import com.google.common.collect.Sets;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.Context;

public class TestCSV {

	static File file = new File("./data/iris.data");
	static Set<String> labels = Sets.newHashSet("Iris-versicolor",
			"Iris-setosa", "Iris-virginica");

	@Test
	public void testLabel() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		reader.mark(10000);
		String[] parts = reader.readLine().split(",");
		int size = parts.length;
		reader.reset();

		CSV csv = CSV.numericColumnCSV(size).setLabelColumn(size - 1)
				.initialize();
		String line;

		while (null != (line = reader.readLine())) {
			List<String> partList = Arrays.asList(line.split(","));
			Context<LinearVector> context = csv.parseRow(partList);
			assertEquals(size - 1, context.getData().size());
			assertNotNull(context.getLabel());
			assertTrue(labels.contains(context.getLabel()));
			assertNull(context.getId());
		}
	}

	@Test
	public void testId() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		reader.mark(10000);
		String[] parts = reader.readLine().split(",");
		int size = parts.length;
		reader.reset();

		CSV csv = CSV.numericColumnCSV(size).setIdColumn(size - 1).initialize();
		String line;

		while (null != (line = reader.readLine())) {
			List<String> partList = Arrays.asList(line.split(","));
			Context<LinearVector> context = csv.parseRow(partList);
			assertEquals(size - 1, context.getData().size());
			assertNotNull(context.getId());
			assertTrue(labels.contains(context.getId()));
			assertNull(context.getLabel());
		}
	}

	@Test
	public void testCategorical() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		reader.mark(10000);
		String[] parts = reader.readLine().split(",");
		int size = parts.length;
		reader.reset();

		CSV csv = CSV.numericColumnCSV(size)
				.setValuesForCategoricalColumn(size - 1, labels).initialize();
		String line;

		while (null != (line = reader.readLine())) {
			List<String> partList = Arrays.asList(line.split(","));
			Context<LinearVector> context = csv.parseRow(partList);
			assertEquals(size + 2, context.getData().size());
			assertNull(context.getId());
			assertNull(context.getLabel());
			assertEquals(1d, context.getData().getValue(4)
					+ context.getData().getValue(5)
					+ context.getData().getValue(6), 0d);
		}
	}
}
